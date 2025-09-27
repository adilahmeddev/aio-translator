use aio_translator_interface::{
    AsyncTranslator, Language, Model, TranslationListOutput, TranslationOutput,
    error::{self, Error},
    prompt::PromptBuilder,
    tokenizer::SentenceTokenizer,
};
use ct2rs::{BatchType, ComputeType, Config, Device, Tokenizer, TranslationOptions};

use interface_model::{
    ModelLoad, ModelRead, ModelSource, ModelWrap, impl_model_helpers, impl_model_load_helpers,
};
use maplit::hashmap;
use regex::Regex;

pub struct SugoiTranslator {
    loaded_models: ModelWrap<ct2rs::Translator<MyTokenizer>>,
    cuda: bool,
    compute_type: ComputeType,
}

fn split_sentences(q: &str, re: &Regex) -> Vec<String> {
    let mut result = Vec::new();
    let mut last = 0;

    for mat in re.find_iter(q) {
        let start = mat.start();
        let end = mat.end();
        if last < start {
            result.push(q[last..start].to_string());
        }
        result.push(q[start..end].to_string());
        last = end;
    }

    if last < q.len() {
        result.push(q[last..].to_string());
    }

    result
}
fn tokenize(queries: &[String]) -> (Vec<String>, Vec<usize>) {
    let mut new_queries: Vec<String> = vec![];
    let mut query_split_sizes: Vec<usize> = vec![];
    let re2 = Regex::new(r"[.。]").unwrap();

    let re = Regex::new(r"(\w[.‥…!?。・]+)").unwrap();

    for q in queries {
        let sentences = split_sentences(&q, &re);
        let mut chunk_queries = vec![];
        for chunk in sentences.chunks(4) {
            let s = chunk.concat();
            let replaced = re2.replace_all(&s, "@").to_string();
            chunk_queries.push(replaced);
        }
        query_split_sizes.push(chunk_queries.len());
        new_queries.extend(chunk_queries);
    }
    (new_queries, query_split_sizes)
}

fn detokenize(queries: Vec<String>, query_split_sizes: Vec<usize>) -> Vec<String> {
    let mut new_translations = vec![];
    let mut i = 0;
    for query_count in query_split_sizes {
        let sentences = &queries[i..i + query_count].join(" ");
        i += query_count;
        let sentences = sentences
            .replace('@', ".")
            .replace('▁', " ")
            .replace("<unk>", "");
        new_translations.push(sentences);
    }
    new_translations
}

impl SugoiTranslator {
    /// single_loaded will only allow one model to be loaded at a time.
    pub fn new(cuda: bool, compute_type: ComputeType) -> Self {
        SugoiTranslator {
            compute_type,
            cuda,
            loaded_models: Default::default(),
        }
    }

    fn pre_tokenize(&self, queries: &[String]) -> Result<(Vec<String>, Vec<usize>), Error> {
        let (queries, query_split_sizes) = tokenize(queries);
        Ok((queries, query_split_sizes))
    }

    fn post_detokenize(
        &self,
        sentences: Vec<String>,
        query_split_sizes: Vec<usize>,
    ) -> anyhow::Result<Vec<String>> {
        Ok(detokenize(sentences, query_split_sizes))
    }
}

#[async_trait::async_trait]
impl AsyncTranslator for SugoiTranslator {
    fn local(&self) -> bool {
        true
    }
    async fn translate(
        &self,
        query: &str,
        _: Option<PromptBuilder>,
        from: Option<Language>,
        to: &Language,
    ) -> anyhow::Result<TranslationOutput> {
        let mut arr = self
            .translate_vec(&vec![query.to_owned()], None, from, to)
            .await?;
        Ok(TranslationOutput {
            text: arr.text.remove(0),
            lang: None,
        })
    }

    async fn translate_vec(
        &self,
        query: &[String],
        _: Option<PromptBuilder>,
        from: Option<Language>,
        to: &Language,
    ) -> anyhow::Result<TranslationListOutput> {
        if let (Some(Language::Japanese), Language::English) = (from, to) {
        } else {
            Err(error::Error::UnknownLanguageGroup(from, to.clone()))?;
        };

        let (query, query_split_sizes) = self.pre_tokenize(query)?;
        let model = self.load().await?;
        let trans = model.translate_batch(
            &query,
            &TranslationOptions {
                batch_type: BatchType::Examples,
                beam_size: 5,
                repetition_penalty: 3.0,
                num_hypotheses: 1,
                replace_unknowns: true,
                disable_unk: true,
                return_alternatives: false,
                ..Default::default()
            },
            None,
        )?;
        Ok(TranslationListOutput {
            text: self
                .post_detokenize(trans.into_iter().map(|v| v.0).collect(), query_split_sizes)?,
            lang: None,
        })
    }
}

pub struct MyTokenizer {
    ja: SentenceTokenizer,
    en: SentenceTokenizer,
}

impl Tokenizer for MyTokenizer {
    fn encode(&self, input: &str) -> anyhow::Result<Vec<String>> {
        self.ja.encode(input)
    }

    fn decode(&self, tokens: Vec<String>) -> anyhow::Result<String> {
        self.en.decode(tokens)
    }
}

#[async_trait::async_trait]
impl ModelLoad for SugoiTranslator {
    impl_model_load_helpers!(loaded_models, ct2rs::Translator<MyTokenizer>);

    async fn reload(&self) -> anyhow::Result<ModelRead<'_, Self::T>> {
        let ja_path = self
            .download_model("spm.ja.nopretok", "spm.ja.nopretok.model")
            .await?;
        let en_path = self
            .download_model("spm.en.nopretok", "spm.en.nopretok.model")
            .await?;

        let model = self.download_model("ja-en", "ja-en/model.bin").await?;

        let model = model.parent().map(|v| v.to_path_buf()).unwrap_or(model);

        let v = ct2rs::Translator::with_tokenizer(
            model,
            MyTokenizer {
                ja: SentenceTokenizer::new(ja_path),
                en: SentenceTokenizer::new(en_path),
            },
            &Config {
                device: match self.cuda {
                    true => Device::CUDA,
                    false => Device::CPU,
                },
                compute_type: self.compute_type,
                ..Default::default()
            },
        )?;
        *self.loaded_models.write().await = Some(v);
        Ok(self.get_model().await.unwrap())
    }
}

impl Model for SugoiTranslator {
    impl_model_helpers!("translator", "sugoi", loaded_models);

    fn models(&self) -> std::collections::HashMap<&'static str, interface_model::ModelSource> {
        hashmap! {
            "ja-en" => ModelSource {
                url: "https://github.com/frederik-uni/aiotranslator/releases/download/sugoi/ja-en.tar.gz",
                hash: "1bb89212e1024e6ad649ed212a4201a524231c46b565819c3112e4c46b38b7ad",
            },
            "spm.en.nopretok" => ModelSource {
                url: "https://github.com/frederik-uni/aiotranslator/releases/download/sugoi/spm.en.nopretok.model",
                hash: "183aad11f5fa48b21fdbbb7a97082d160b86cbcc4f9dc5e61d0eebd48390d36a"
            },
            "spm.ja.nopretok" => ModelSource {
                url: "https://github.com/frederik-uni/aiotranslator/releases/download/sugoi/spm.ja.nopretok.model",
                hash: "1bff3529a8e0bd898f00707a4e36dc16540d84112cc8a4a14462c0099e4aab9d"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use env_logger::Env;

    use super::*;

    #[tokio::test]
    async fn test_load() {
        let sugoi = SugoiTranslator::new(false, ComputeType::DEFAULT);
        assert!(sugoi.load().await.is_ok());
        assert!(sugoi.loaded().await);
    }

    #[tokio::test]
    async fn test_translate() {
        env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();
        let sugoi = SugoiTranslator::new(false, ComputeType::DEFAULT);
        let input_ja = vec![
            "明日は雨が降るかもしれません。".to_string(),
            "彼はその問題について深く考えている。".to_string(),
            "このソフトウェアは非常に使いやすいです。".to_string(),
        ];

        let out = sugoi
            .translate_vec(
                &input_ja,
                None,
                Some(Language::Japanese),
                &Language::English,
            )
            .await
            .expect("Translation failed");
        assert_eq!(
            out.text,
            vec![
                "It might rain tomorrow.".to_owned(),
                "He's thinking deeply about the problem.".to_owned(),
                "This software is very easy to use.".to_owned()
            ]
        );
    }
}
