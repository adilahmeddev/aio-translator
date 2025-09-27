use aio_translator_interface::{
    AsyncTranslator, Language, TranslationListOutput, TranslationOutput, prompt::PromptBuilder,
};

pub struct OriginalTranslator {}

impl OriginalTranslator {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait::async_trait]
impl AsyncTranslator for OriginalTranslator {
    fn local(&self) -> bool {
        false
    }

    async fn translate(
        &self,
        input: &str,
        _: Option<PromptBuilder>,
        _: Option<Language>,
        _: &Language,
    ) -> anyhow::Result<TranslationOutput> {
        Ok(TranslationOutput {
            text: input.to_owned(),
            lang: None,
        })
    }

    async fn translate_vec(
        &self,
        items: &[String],
        _: Option<PromptBuilder>,
        _: Option<Language>,
        _: &Language,
    ) -> anyhow::Result<TranslationListOutput> {
        Ok(TranslationListOutput {
            text: items.to_vec(),
            lang: None,
        })
    }
}
