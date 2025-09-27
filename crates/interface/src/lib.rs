pub mod error;
pub mod prompt;
pub mod tokenizer;

use crate::prompt::PromptBuilder;
use aio_translator_lang_generator::generate_language;
pub use interface_model::Model;

generate_language!();

pub trait Detector {
    fn detect_language(&self, text: &str) -> Option<Language>;
}

#[async_trait::async_trait]
pub trait AsyncTranslator: Send + Sync {
    fn local(&self) -> bool;
    async fn translate(
        &self,
        query: &str,
        context: Option<PromptBuilder>,
        from: Option<Language>,
        to: &Language,
    ) -> anyhow::Result<TranslationOutput>;

    async fn translate_vec(
        &self,
        query: &[String],
        context: Option<PromptBuilder>,
        from: Option<Language>,
        to: &Language,
    ) -> anyhow::Result<TranslationListOutput>;
}

/// Translation Result containing the translation and the language
#[derive(Clone, Debug)]
pub struct TranslationOutput {
    /// Translation
    pub text: String,
    /// Text language
    pub lang: Option<Language>,
}

/// Translation Result containing the translation and the language
#[derive(Clone, Debug)]
pub struct TranslationListOutput {
    /// Translation
    pub text: Vec<String>,
    /// Text language
    pub lang: Option<Language>,
}
