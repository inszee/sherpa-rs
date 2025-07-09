use crate::{
    get_default_provider,
    utils::{cstr_to_string, cstring_from_str},
    zipformer::ZipFormerConfig,
};
use eyre::{bail, Result};
use serde::{Deserialize, Serialize};
use std::{ffi::CStr, mem};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct OnlineRecognizerResult {
    pub text: String,
    pub tokens: Vec<String>,
    pub timestamps: Vec<f64>,
    pub segment: u64,
}

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
        let paraformer = unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineParaformerModelConfig {
                encoder: mem::zeroed::<_>(),
                decoder: mem::zeroed::<_>(),
            }
        };
        let ctc = unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineZipformer2CtcModelConfig {
                model: mem::zeroed::<_>(),
            }
        };

        // Offline model config
        let model_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineModelConfig {
                transducer: transcuder_config,
                paraformer: paraformer,
                zipformer2_ctc: ctc,
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

        let feat_config = sherpa_rs_sys::SherpaOnnxFeatureConfig {
            sample_rate: 16000,
            feature_dim: 80,
        };

        let ctc_decode = unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineCtcFstDecoderConfig {
                graph: mem::zeroed::<_>(),
                max_active: 3000,
            }
        };

        let hr = unsafe {
            sherpa_rs_sys::SherpaOnnxHomophoneReplacerConfig {
                dict_dir: mem::zeroed::<_>(),
                lexicon: mem::zeroed::<_>(),
                rule_fsts: mem::zeroed::<_>(),
            }
        };
        // Recognizer config
        let recognizer_config = unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineRecognizerConfig {
                model_config,
                decoding_method: decoding_method_ptr.as_ptr(),
                // NULLs
                feat_config: feat_config,
                max_active_paths: 4,
                enable_endpoint: 1,
                rule1_min_trailing_silence: 2.4,
                rule2_min_trailing_silence: 1.2,
                rule3_min_utterance_length: 20.0,
                blank_penalty: 0.0,
                hotwords_file: mem::zeroed::<_>(),
                hotwords_score: 1.5,
                ctc_fst_decoder_config: ctc_decode,
                rule_fars: mem::zeroed::<_>(),
                rule_fsts: mem::zeroed::<_>(),
                hotwords_buf: mem::zeroed::<_>(),
                hotwords_buf_size: 10,
                hr: hr,
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

    pub fn decode(&mut self, sample_rate: u32, samples: Vec<f32>) -> OnlineRecognizerResult {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len().try_into().unwrap(),
            );
            while sherpa_rs_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer, self.stream) == 1 {
                sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer, self.stream);
            }

            let result_ptr =
                sherpa_rs_sys::SherpaOnnxGetOnlineStreamResultAsJson(self.recognizer, self.stream);

            let text = if !result_ptr.is_null() {
                // let raw_result = result_ptr.read();
                let json_string = CStr::from_ptr(result_ptr)
                    .to_string_lossy() // allows malformed UTF-8 (like Dart's `allowMalformed`)
                    .into_owned();
                let json_result = serde_json::from_str::<OnlineRecognizerResult>(&json_string).ok();
                sherpa_rs_sys::SherpaOnnxDestroyOnlineStreamResultJson(result_ptr);
                if json_result.is_some() {
                    json_result.unwrap()
                } else {
                    OnlineRecognizerResult::default()
                }
            } else {
                OnlineRecognizerResult::default()
            };

            if sherpa_rs_sys::SherpaOnnxOnlineStreamIsEndpoint(self.recognizer, self.stream) == 1 {
                sherpa_rs_sys::SherpaOnnxOnlineStreamReset(self.recognizer, self.stream);
            }
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
