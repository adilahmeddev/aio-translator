use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

use aio_translator_interface::{
    AsyncTranslator, Language, TranslationListOutput, TranslationOutput, prompt::PromptBuilder,
};
use async_trait::async_trait;
use tokio::sync::{Mutex, Semaphore, SemaphorePermit};

pub struct RateLimiter<T: AsyncTranslator> {
    t: T,
    /// Max concurrency limiter
    semaphore: Option<Arc<Semaphore>>,
    ///  rate limiter queue
    timestamps: Option<Arc<Mutex<VecDeque<Instant>>>>,
    /// Max requests within time window
    max_requests: usize,
    /// Time window for rate limiting
    window: Duration,
}

impl<T: AsyncTranslator> RateLimiter<T> {
    /// Create a new RateLimiter wrapper
    /// - `concurrent_limit`: Some(n) to allow up to n concurrent requests
    /// - `max_requests` and `window`: allow at most `max_requests` within `window`
    pub fn new(
        t: T,
        concurrent_limit: Option<usize>,
        max_requests: Option<(usize, Duration)>,
    ) -> Self {
        let semaphore = concurrent_limit.map(|n| Arc::new(Semaphore::new(n)));
        let (max_requests, window, timestamps) = if let Some((n, d)) = max_requests {
            (n, d, Some(Arc::new(Mutex::new(VecDeque::with_capacity(n)))))
        } else {
            (usize::MAX, Duration::ZERO, None)
        };

        Self {
            t,
            semaphore,
            timestamps,
            max_requests,
            window,
        }
    }

    /// Internal function to enforce concurrency + rate limits
    async fn acquire(&self) -> Option<SemaphorePermit<'_>> {
        if let Some(timestamps) = &self.timestamps {
            loop {
                let mut queue = timestamps.lock().await;
                let now = Instant::now();

                while let Some(&front) = queue.front() {
                    if now.duration_since(front) > self.window {
                        queue.pop_front();
                    } else {
                        break;
                    }
                }

                if queue.len() < self.max_requests {
                    queue.push_back(now);
                    break;
                }

                if let Some(&oldest) = queue.front() {
                    let wait_time = self.window.saturating_sub(now.duration_since(oldest));
                    drop(queue);
                    tokio::time::sleep(wait_time).await;
                }
            }
        }

        let _permit = if let Some(sem) = &self.semaphore {
            Some(sem.acquire().await.expect("Semaphore closed"))
        } else {
            None
        };

        _permit
    }
}

#[async_trait]
impl<T: AsyncTranslator + Send + Sync> AsyncTranslator for RateLimiter<T> {
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
        let permit = self.acquire().await;
        let r = self.t.translate(query, context, from, to).await;
        drop(permit);
        r
    }

    async fn translate_vec(
        &self,
        query: &[String],
        context: Option<PromptBuilder>,
        from: Option<Language>,
        to: &Language,
    ) -> anyhow::Result<TranslationListOutput> {
        let permit = self.acquire().await;
        let r = self.t.translate_vec(query, context, from, to).await;
        drop(permit);
        r
    }
}
