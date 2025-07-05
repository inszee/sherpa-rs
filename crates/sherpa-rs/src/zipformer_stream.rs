use crate::{
    get_default_provider,
    utils::{cstr_to_string, cstring_from_str},
    zipformer::ZipFormerConfig,
};
use eyre::{bail, Result};
use std::mem;

pub struct ZipFormerStream {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
    stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream,
}

impl ZipFormerStream {
    pub fn new(config: ZipFormerConfig) -> Result<Self> {
        // Zipformer config
        let decoder_ptr = cstring_from_str(&config.decoder);
        let encoder_ptr = cstring_from_str(&config.encoder);
        let joiner_ptr = cstring_from_str(&config.joiner);
        let provider_ptr = cstring_from_str(&config.provider.unwrap_or(get_default_provider()));
        let tokens_ptr = cstring_from_str(&config.tokens);
        let decoding_method_ptr = cstring_from_str("greedy_search");

        let transcuder_config = sherpa_rs_sys::SherpaOnnxOnlineTransducerModelConfig {
            decoder: decoder_ptr.as_ptr(),
            encoder: encoder_ptr.as_ptr(),
            joiner: joiner_ptr.as_ptr(),
        };
        // Offline model config
        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineModelConfig {
                transducer: transcuder_config,
                paraformer: mem::zeroed::<_>(),
                zipformer2_ctc: mem::zeroed::<_>(),
                tokens: tokens_ptr.as_ptr(),
                num_threads: config.num_threads.unwrap_or(1),
                provider: provider_ptr.as_ptr(),
                debug: config.debug.into(),
                model_type: mem::zeroed::<_>(),
                modeling_unit: mem::zeroed::<_>(),
                bpe_vocab: mem::zeroed::<_>(),
                tokens_buf: mem::zeroed::<_>(),
                tokens_buf_size: 0,
            }
        };
        // Recognizer config
        let recognizer_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineRecognizerConfig {
                model_config,
                decoding_method: decoding_method_ptr.as_ptr(),
                // NULLs
                feat_config: mem::zeroed::<_>(),
                max_active_paths: mem::zeroed::<_>(),
                enable_endpoint: mem::zeroed::<_>(),
                rule1_min_trailing_silence: mem::zeroed::<_>(),
                rule2_min_trailing_silence: mem::zeroed::<_>(),
                rule3_min_utterance_length: mem::zeroed::<_>(),
                blank_penalty: mem::zeroed::<_>(),
                hotwords_file: mem::zeroed::<_>(),
                hotwords_score: mem::zeroed::<_>(),
                ctc_fst_decoder_config: mem::zeroed::<_>(),
                rule_fars: mem::zeroed::<_>(),
                rule_fsts: mem::zeroed::<_>(),
                hotwords_buf: mem::zeroed::<_>(),
                hotwords_buf_size: mem::zeroed::<_>(),
                hr: mem::zeroed::<_>(),
            }
        };

        let recognizer =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineRecognizer(&recognizer_config) };

        if recognizer.is_null() {
            bail!("Failed to create recognizer");
        }
        let stream = unsafe { sherpa_rs_sys::SherpaOnnxCreateOnlineStream(recognizer) };
        if stream.is_null() {
            bail!("Failed to create stream");
        }
        Ok(Self { recognizer, stream })
    }

    pub fn decode(&mut self, sample_rate: u32, samples: Vec<f32>) -> String {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len().try_into().unwrap(),
            );
            while sherpa_rs_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer, self.stream) > 0 {
                sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer, self.stream);
            }

            let result_ptr =
                sherpa_rs_sys::SherpaOnnxGetOnlineStreamResult(self.recognizer, self.stream);
            // let raw_result = result_ptr.read();
            // let text = cstr_to_string(raw_result.text as _);
            // // Free
            // sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            let text = if !result_ptr.is_null() {
                let raw_result = result_ptr.read();
                let text = cstr_to_string(raw_result.text as _);
                sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizerResult(result_ptr);
                text
            } else {
                String::new()
            };
            text
        }
    }
}

unsafe impl Send for ZipFormerStream {}
unsafe impl Sync for ZipFormerStream {}

impl Drop for ZipFormerStream {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.stream);
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
    }
}
