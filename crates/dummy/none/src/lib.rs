use aio_translator_interface::{
    AsyncTranslator, Language, TranslationListOutput, TranslationOutput, prompt::PromptBuilder,
};

pub struct NoneTranslator {}

impl NoneTranslator {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait::async_trait]
impl AsyncTranslator for NoneTranslator {
    fn local(&self) -> bool {
        false
    }

    async fn translate(
        &self,
        _: &str,
        _: Option<PromptBuilder>,
        _: Option<Language>,
        _: &Language,
    ) -> anyhow::Result<TranslationOutput> {
        Ok(TranslationOutput {
            text: Default::default(),
            lang: None,
        })
    }

    async fn translate_vec(
        &self,
        _: &[String],
        _: Option<PromptBuilder>,
        _: Option<Language>,
        _: &Language,
    ) -> anyhow::Result<TranslationListOutput> {
        Ok(TranslationListOutput {
            text: vec![],
            lang: None,
        })
    }
}
