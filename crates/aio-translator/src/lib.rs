mod rate_limit;
mod style_transfer;

pub use aio_translator_interface::{
    AsyncTranslator, Detector, Language, Model, TranslationListOutput, TranslationOutput,
    error::ApiError, error::Error, prompt::PromptBuilder,
};

pub use aio_translator_baidu::BaiduTranslator;
pub use aio_translator_caiyun::CaiyunTranslator;
pub use aio_translator_deepl::DeeplTranslator;
pub use aio_translator_google::GoogleTranslator;
pub use aio_translator_jparacrawl::JParaCrawlTranslator;
pub use aio_translator_jparacrawl::Size as JParaCrawlSize;
pub use aio_translator_langid::LangIdDetector;
#[cfg(feature = "lingua")]
pub use aio_translator_lingua::LinguaDetector;
pub use aio_translator_m2m100::M2M100Translator;
pub use aio_translator_m2m100::Size as M2M100Size;
pub use aio_translator_mbart50::MBart50Translator;
pub use aio_translator_mymemory::MyMemoryTranslator;
pub use aio_translator_nllb::NLLBTranslator;
pub use aio_translator_nllb::Size as NLLBSize;
pub use aio_translator_none::NoneTranslator;
pub use aio_translator_original::OriginalTranslator;
pub use aio_translator_papago::PapagoTranslator;
pub use aio_translator_sugoi::SugoiTranslator;
#[cfg(feature = "whatlang")]
pub use aio_translator_whatlang::WhatLangDetector;
pub use aio_translator_youdao::YoudaoTranslator;
pub use ct2rs::ComputeType;
pub mod wrapper {
    pub use crate::rate_limit::RateLimiter;
    pub use crate::style_transfer::StyleTransfer;
}

pub use style_transfer::is_valuable_text;
