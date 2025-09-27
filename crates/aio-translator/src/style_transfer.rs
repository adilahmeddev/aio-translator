use aio_translator_interface::{
    AsyncTranslator, Language, TranslationListOutput, TranslationOutput, prompt::PromptBuilder,
};
use async_trait::async_trait;
use fancy_regex::Regex;
use unicode_general_category::{GeneralCategory, get_general_category};

pub struct StyleTransfer<T: AsyncTranslator> {
    t: T,
}

#[async_trait]
impl<T: AsyncTranslator + Send + Sync> AsyncTranslator for StyleTransfer<T> {
    fn local(&self) -> bool {
        self.t.local()
    }
    async fn translate(
        &self,
        query: &str,
        context: Option<PromptBuilder>,
        from: Option<Language>,
        to: &Language,
    ) -> anyhow::Result<TranslationOutput> {
        if from == Some(*to) {
            return Ok(TranslationOutput {
                text: query.to_owned(),
                lang: from,
            });
        }
        let mut trans = self.t.translate(query, context, from, to).await?;
        if is_valuable_text(&trans.text) {
            trans.text = clean_translation_output(query, &trans.text, *to);
        }
        Ok(trans)
    }

    async fn translate_vec(
        &self,
        query: &[String],
        context: Option<PromptBuilder>,
        from: Option<Language>,
        to: &Language,
    ) -> anyhow::Result<TranslationListOutput> {
        if from == Some(*to) {
            return Ok(TranslationListOutput {
                text: query.to_owned(),
                lang: from,
            });
        }
        let mut trans = self.t.translate_vec(query, context, from, to).await?;
        trans.text = query
            .iter()
            .zip(trans.text)
            .map(|(query, trans)| match is_valuable_text(&trans) {
                true => clean_translation_output(query, &trans, *to),
                false => query.to_owned(),
            })
            .collect();
        Ok(trans)
    }
}

fn clean_translation_output(query: &str, trans: &str, to_lang: Language) -> String {
    let trans = trans.split_whitespace().collect::<Vec<_>>().join(" ");
    //trans = re.sub(r'(?<![.,;!?])([.,;!?])(?=\w)', r'\1 ', trans);
    let trans = Regex::new(r"([^\.,;!?\s])([.,;!?])(?=\w)")
        .unwrap()
        .replace_all(&trans, "$1$2 ");
    // trans = re.sub(r'([.,;!?])\s+(?=[.,;!?]|$)', r'\1', trans);
    let mut trans = Regex::new(r"([.,;!?])\s+(?=[.,;!?]|$)")
        .unwrap()
        .replace_all(&trans, "$1")
        .to_string();

    if to_lang != Language::Arabic {
        // trans = re.sub(r'(?<=[.,;!?\w])\s+([.,;!?])', r'\1', trans);
        let t = Regex::new(r"([.,;!?\w])\s+([.,;!?])")
            .unwrap()
            .replace_all(&trans, "$1$2");
        // trans = re.sub(r'((?:\s|^)\.+)\s+(?=\w)', r'\1', trans);
        trans = Regex::new(r"((?:\s|^)\.+)\s+(?=\w)")
            .unwrap()
            .replace_all(&t, "$1")
            .to_string();
    }

    let seq = repeating_sequence(&trans.to_lowercase());
    if seq.len() < query.len() && trans.len() / 2 > seq.len() {
        let trans_ = seq.repeat(1.max(query.chars().count() / seq.chars().count()));

        trans = query
            .chars()
            .zip(trans_.chars())
            .map(|(s, t)| match s.is_uppercase() {
                true => t.to_uppercase().next().unwrap(),
                false => t,
            })
            .collect::<String>();
    }
    let mut trans = trans.to_string();
    if to_lang == Language::Arabic {
        trans = arabic_reshaper::arabic_reshape(&trans);
    }
    trans
}

fn repeating_sequence(ss: &str) -> String {
    let s = ss.chars().collect::<Vec<_>>();
    let len = s.len();
    for i in 1..=(len / 2) {
        let seq = &s[..i];
        let repeats = len / i;
        let remainder = len % i;
        let candidate = format!(
            "{}{}",
            seq.iter().collect::<String>().repeat(repeats),
            seq[..remainder].iter().collect::<String>()
        );

        if candidate == ss {
            return seq.iter().collect::<String>();
        }
    }
    return ss.to_owned();
}

fn is_valuable_char(ch: char) -> bool {
    !is_punctuation(ch) && !is_control(ch) && !is_whitespace(ch) && !ch.is_numeric()
}

fn is_whitespace(ch: char) -> bool {
    if ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch as u32 == 0 {
        return true;
    }
    get_general_category(ch) == GeneralCategory::SpaceSeparator
}
fn is_punctuation(ch: char) -> bool {
    let cp = ch as u32;

    if (cp >= 33 && cp <= 47)
        || (cp >= 58 && cp <= 64)
        || (cp >= 91 && cp <= 96)
        || (cp >= 123 && cp <= 126)
    {
        return true;
    }

    matches!(
        get_general_category(ch),
        GeneralCategory::ClosePunctuation
            | GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::OtherPunctuation
    )
}
fn is_control(ch: char) -> bool {
    if ch == '\t' || ch == '\n' || ch == '\r' {
        return false;
    }
    matches!(
        get_general_category(ch),
        GeneralCategory::Control | GeneralCategory::Format
    )
}

pub fn is_valuable_text(text: &str) -> bool {
    text.chars().any(|v| is_valuable_char(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_translation_output_basic() {
        let query = "Hello,world!How are you?";
        let trans = "Hello,world!How are you?";
        let result = clean_translation_output(query, trans, Language::English);
        assert_eq!(result, "Hello, world! How are you?");
    }

    #[test]
    fn test_clean_translation_output_repeating_seq() {
        let query = "AbAbAbAbAbAbAbAbAb";
        let trans = "cdcdcdcdcdcdcdcdcd";
        let result = clean_translation_output(query, trans, Language::English);
        assert_eq!(result, "CdCdCdCdCdCdCdCdCd");
    }
}
