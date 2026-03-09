package com.braintumor.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.Executor;

/**
 * Enables @Async for AiInferenceService.
 * AI inference runs on a dedicated thread pool so it doesn't block HTTP threads.
 */
@Configuration
@EnableAsync
public class AsyncConfig {

    @Bean(name = "aiExecutor")
    public Executor aiExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(2);     // 2 concurrent inference jobs
        executor.setMaxPoolSize(4);
        executor.setQueueCapacity(20);
        executor.setThreadNamePrefix("AI-Inference-");
        executor.initialize();
        return executor;
    }
}
