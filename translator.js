import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2'
import { transliterate } from './transliterate.js'

// Setup transformers-js env to force remote model resolving
env.allowLocalModels = false
env.allowRemoteModels = true
env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/'

// Configure ORT
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/'

export const MODEL_CONFIGS = {
  'en-indic-200m': {
    direction: 'en-indic',
    repoId: 'hari31416/indictrans2-en-indic-dist-200M-ONNX',
    label: 'en-indic (200M Distilled)'
  },
  'en-indic-1b': {
    direction: 'en-indic',
    repoId: 'hari31416/indictrans2-en-indic-1B-ONNX',
    label: 'en-indic (1B Full)'
  },
  'indic-en-200m': {
    direction: 'indic-en',
    repoId: 'hari31416/indictrans2-indic-en-dist-200M-ONNX',
    label: 'indic-en (200M Distilled)'
  },
  'indic-en-1b': {
    direction: 'indic-en',
    repoId: 'hari31416/indictrans2-indic-en-1B-ONNX',
    label: 'indic-en (1B Full)'
  },
  'indic-indic-320m': {
    direction: 'indic-indic',
    repoId: 'hari31416/indictrans2-indic-indic-dist-320M-ONNX',
    label: 'indic-indic (320M)'
  },
  'indic-indic-1b': {
    direction: 'indic-indic',
    repoId: 'hari31416/indictrans2-indic-indic-1B-ONNX',
    label: 'indic-indic (1B)'
  }
}

// App state
let currentSessions = null
let srcTokenizer = null
let tgtTokenizer = null
let tokenizerMeta = null
let generationConfig = null

export function isModelLoaded() {
  return currentSessions !== null
}

const ONNX_GRAPH_FILES = [
  'encoder_model.onnx',
  'decoder_model.onnx',
  'decoder_with_past_model.onnx',
]

const EXTERNAL_DATA_CANDIDATES = [
  'encoder_model.onnx.data',
  'decoder_shared.onnx.data',
  'decoder_model.onnx.data',
  'decoder_with_past_model.onnx.data',
]

async function fetchWithCache(url) {
  if (typeof window.caches === 'undefined') {
    return fetch(url)
  }
  const cacheName = 'indictrans2-onnx-cache'
  const cache = await caches.open(cacheName)
  const cachedResponse = await cache.match(url)
  if (cachedResponse) {
    return cachedResponse
  }
  const response = await fetch(url)
  if (response.ok) {
    try {
      await cache.put(url, response.clone())
    } catch (e) {
      console.warn('Failed to store in cache:', e)
    }
  }
  return response
}

async function fetchWithProgress(url, progressCallback) {
  const cacheName = 'indictrans2-onnx-cache'
  const useCache = typeof window.caches !== 'undefined'
  
  if (useCache) {
    const cache = await caches.open(cacheName)
    const cachedResponse = await cache.match(url)
    if (cachedResponse) {
      const contentLength = cachedResponse.headers.get('content-length')
      const total = contentLength ? parseInt(contentLength, 10) : 0
      
      const reader = cachedResponse.body.getReader()
      const chunks = []
      let loaded = 0
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        chunks.push(value)
        loaded += value.length
        if (total) {
          progressCallback(Math.round((loaded / total) * 100))
        }
      }
      
      const all = new Uint8Array(loaded)
      let offset = 0
      for (const chunk of chunks) {
        all.set(chunk, offset)
        offset += chunk.length
      }
      progressCallback(100)
      return all.buffer
    }
  }

  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to fetch file: ${response.statusText}`)
  }
  
  const contentLength = response.headers.get('content-length')
  if (!contentLength) {
    const buffer = await response.arrayBuffer()
    if (useCache) {
      try {
        const cache = await caches.open(cacheName)
        await cache.put(url, new Response(buffer))
      } catch (e) {
        console.warn('Failed to store in cache:', e)
      }
    }
    progressCallback(100)
    return buffer
  }

  const total = parseInt(contentLength, 10)
  let loaded = 0
  const reader = response.body.getReader()
  const chunks = []

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.length
    progressCallback(Math.round((loaded / total) * 100))
  }

  const all = new Uint8Array(loaded)
  let offset = 0
  for (const chunk of chunks) {
    all.set(chunk, offset)
    offset += chunk.length
  }
  
  if (useCache) {
    try {
      const cache = await caches.open(cacheName)
      await cache.put(url, new Response(all.buffer, {
        headers: {
          'content-length': loaded.toString(),
          'content-type': 'application/octet-stream'
        }
      }))
    } catch (e) {
      console.warn('Failed to save to cache:', e)
    }
  }
  
  return all.buffer
}

async function loadTokenizerFromUrl(url, modelName) {
  const originalFetch = window.fetch
  const response = await fetchWithCache(url)
  const data = await response.json()

  window.fetch = async (fetchUrl, options) => {
    const urlStr = fetchUrl.toString()
    if (urlStr.endsWith('tokenizer.json')) {
      return new Response(JSON.stringify(data), { headers: { 'Content-Type': 'application/json' } })
    }
    if (urlStr.endsWith('tokenizer_config.json')) {
      return new Response(JSON.stringify({ tokenizer_class: 'PreTrainedTokenizerFast' }), { headers: { 'Content-Type': 'application/json' } })
    }
    return originalFetch(fetchUrl, options)
  }

  const tok = await AutoTokenizer.from_pretrained(modelName)
  window.fetch = originalFetch // restore fetch
  return tok
}

function getPastFeed(prevOutputs, numLayers) {
  const feed = {}
  for (let i = 0; i < numLayers; i++) {
    feed[`past_key_values.${i}.decoder.key`] = prevOutputs[`present.${i}.decoder.key`]
    feed[`past_key_values.${i}.decoder.value`] = prevOutputs[`present.${i}.decoder.value`]
    feed[`past_key_values.${i}.encoder.key`] = prevOutputs[`present.${i}.encoder.key`]
    feed[`past_key_values.${i}.encoder.value`] = prevOutputs[`present.${i}.encoder.value`]
  }
  return feed
}

async function probeExternalDataUrls(baseUrl) {
  const root = baseUrl.replace(/\/$/, '')
  const found = []
  for (const name of EXTERNAL_DATA_CANDIDATES) {
    try {
      const res = await fetch(`${root}/${name}`, { method: 'HEAD' })
      if (res.ok) {
        found.push(name)
      }
    } catch {
      // ignore unreachable sidecars
    }
  }
  return found
}

function buildExternalDataFromBuffers(buffers) {
  const externalData = []
  for (const [name, bytes] of buffers) {
    externalData.push({ path: name, data: bytes })
    if (!name.startsWith('./')) {
      externalData.push({ path: `./${name}`, data: bytes })
    }
  }
  return externalData
}

async function fetchExternalDataSidecars(baseUrl, onProgress) {
  const sidecarNames = await probeExternalDataUrls(baseUrl)
  if (sidecarNames.length === 0) {
    return []
  }

  const buffers = new Map()
  for (const name of sidecarNames) {
    const progressId = `data-${name.replace(/\./g, '-')}`
    const buffer = await fetchWithProgress(`${baseUrl}/${name}`, (p) => {
      onProgress(progressId, name, p)
    })
    buffers.set(name, new Uint8Array(buffer))
  }

  ort.env.wasm.numThreads = 1
  return buildExternalDataFromBuffers(buffers)
}

async function createSessionFromBuffer(modelBuffer, ortOptions, externalData) {
  const options = { ...ortOptions }
  if (externalData.length > 0) {
    options.externalData = externalData
  }
  return ort.InferenceSession.create(modelBuffer, options)
}

export async function loadModel(configKey, precision, provider, onProgress) {
  const config = MODEL_CONFIGS[configKey]
  const isFp32 = (precision === 'fp32')
  const suffix = isFp32 ? '' : `-${precision}`
  const repoId = `${config.repoId}${suffix}`
  const baseUrl = `https://huggingface.co/${repoId}/resolve/main`

  // 1. Fetch configs and metadata
  onProgress('meta', 'Tokenizer Meta & Configurations', 10)
  const metaResponse = await fetchWithCache(`${baseUrl}/tokenizer_meta.json`)
  tokenizerMeta = await metaResponse.json()
  onProgress('meta', 'Tokenizer Meta & Configurations', 50)

  const genConfigResponse = await fetchWithCache(`${baseUrl}/generation_config.json`)
  generationConfig = await genConfigResponse.json()
  onProgress('meta', 'Tokenizer Meta & Configurations', 100)

  // 2. Fetch tokenizers
  onProgress('tok-src', 'Source Tokenizer', 10)
  srcTokenizer = await loadTokenizerFromUrl(`${baseUrl}/tokenizer_src.json`, `${configKey}-${precision}-src`)
  onProgress('tok-src', 'Source Tokenizer', 100)

  onProgress('tok-tgt', 'Target Tokenizer', 10)
  tgtTokenizer = await loadTokenizerFromUrl(`${baseUrl}/tokenizer_tgt.json`, `${configKey}-${precision}-tgt`)
  onProgress('tok-tgt', 'Target Tokenizer', 100)

  // 3. Fetch weight sidecars (optimized bundles externalize weights to .onnx.data)
  onProgress('sidecars', 'Weight sidecars', 0)
  const externalData = await fetchExternalDataSidecars(baseUrl, onProgress)
  onProgress('sidecars', 'Weight sidecars', 100)

  // 4. Load ONNX sessions (graph protos + externalData for WASM)
  const ortOptions = {
    executionProviders: [provider, 'wasm'],
  }

  const sessions = {}
  const progressIds = {
    'encoder_model.onnx': ['enc', 'Encoder Model'],
    'decoder_model.onnx': ['dec', 'Decoder Model'],
    'decoder_with_past_model.onnx': ['dec_past', 'Decoder with Past'],
  }

  for (const graphName of ONNX_GRAPH_FILES) {
    const [progressId, label] = progressIds[graphName]
    onProgress(progressId, label, 0)
    const graphBuffer = await fetchWithProgress(`${baseUrl}/${graphName}`, (p) => {
      onProgress(progressId, label, p)
    })
    sessions[graphName] = await createSessionFromBuffer(graphBuffer, ortOptions, externalData)
    onProgress(progressId, label, 100)
  }

  const decSession = sessions['decoder_model.onnx']
  const numLayers = (decSession.outputNames.length - 1) / 4

  currentSessions = {
    enc: sessions['encoder_model.onnx'],
    dec: decSession,
    decPast: sessions['decoder_with_past_model.onnx'],
    numLayers,
  }
}

export function unloadModel() {
  currentSessions = null
  srcTokenizer = null
  tgtTokenizer = null
  tokenizerMeta = null
  generationConfig = null
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1)
}

export async function translate(text, srcLang, tgtLang, onStep) {
  if (!currentSessions) {
    throw new Error('No model loaded')
  }

  const startTime = performance.now()

  // 1. Encode source
  let processedText = text
  if (srcLang !== 'eng_Latn') {
    processedText = transliterate(text, srcLang, 'hin_Deva')
  }

  // We tokenize the language tags and text separately to work around a Transformers.js
  // BPE pre-tokenization bug where prefix spaces are dropped after splitting added tokens.
  const srcLangRes = await srcTokenizer(srcLang, { add_special_tokens: false })
  const srcLangId = Number(srcLangRes.input_ids.data[0])

  const tgtLangRes = await srcTokenizer(tgtLang, { add_special_tokens: false })
  const tgtLangId = Number(tgtLangRes.input_ids.data[0])

  const preparedText = processedText.startsWith(' ') ? processedText : ' ' + processedText
  const textRes = await srcTokenizer(preparedText)
  
  const textIds = Array.from(textRes.input_ids.data).map(Number)
  const safeInputIds = [srcLangId, tgtLangId, ...textIds].map(id => {
    return id < tokenizerMeta.src_dict_size ? id : Number(tokenizerMeta.unk_id)
  })

  const inputIdsTensor = new ort.Tensor('int64', BigInt64Array.from(safeInputIds.map(BigInt)), [1, safeInputIds.length])

  // For attention mask: the language tags are not masked, so we prepend 1s for them.
  const textMaskArray = Array.from(textRes.attention_mask.data).map(Number)
  const attnMaskArray = [1, 1, ...textMaskArray]
  const attnMaskTensor = new ort.Tensor('int64', BigInt64Array.from(attnMaskArray.map(BigInt)), [1, attnMaskArray.length])

  // Run Encoder
  const encOut = await currentSessions.enc.run({
    input_ids: inputIdsTensor,
    attention_mask: attnMaskTensor
  })
  const encHiddenState = encOut.last_hidden_state

  // Decode config parameters
  const decoderStartId = BigInt(generationConfig.decoder_start_token_id || 2)
  const eosId = BigInt(generationConfig.eos_token_id || 2)

  let decoderInputIds = new ort.Tensor('int64', BigInt64Array.from([decoderStartId]), [1, 1])
  const outputIds = [Number(decoderStartId)]
  let pastOutputs = null

  const maxNewTokens = 128
  let totalTokens = 0
  let ttftTime = null

  for (let step = 0; step < maxNewTokens; step++) {
    let decOut
    if (step === 0) {
      decOut = await currentSessions.dec.run({
        input_ids: decoderInputIds,
        encoder_hidden_states: encHiddenState,
        encoder_attention_mask: attnMaskTensor
      })
    } else {
      const feeds = {
        input_ids: decoderInputIds,
        encoder_attention_mask: attnMaskTensor,
        ...getPastFeed(pastOutputs, currentSessions.numLayers)
      }
      decOut = await currentSessions.decPast.run(feeds)
    }

    if (step === 0) {
      ttftTime = performance.now() - startTime
    }

    const logits = decOut['logits']
    pastOutputs = decOut

    // Argmax last token logits
    const dims = logits.dims
    const seqLen = dims[1]
    const vocabSize = dims[2]
    const offset = (seqLen - 1) * vocabSize
    const logitsData = logits.data

    let maxVal = -Infinity
    let nextId = 0
    for (let v = 0; v < vocabSize; v++) {
      const val = logitsData[offset + v]
      if (val > maxVal) {
        maxVal = val
        nextId = v
      }
    }

    outputIds.push(nextId)
    totalTokens++

    if (onStep) {
      onStep(step, totalTokens, ttftTime)
    }

    if (BigInt(nextId) === eosId) {
      break
    }

    decoderInputIds = new ort.Tensor('int64', BigInt64Array.from([BigInt(nextId)]), [1, 1])
  }

  // 2. Decode outputs
  const tgtDictSize = tokenizerMeta.tgt_dict_size
  const safeOutputIds = outputIds.map(id => (id < tgtDictSize ? id : Number(tokenizerMeta.unk_id)))
  const decodedText = await tgtTokenizer.decode(safeOutputIds, { skip_special_tokens: true })

  let finalOutput = decodedText
  if (tgtLang !== 'eng_Latn') {
    finalOutput = transliterate(decodedText, 'hin_Deva', tgtLang)
  }

  const totalTimeMs = performance.now() - startTime

  return {
    translation: finalOutput,
    totalTokens,
    ttftTime,
    totalTimeMs
  }
}
