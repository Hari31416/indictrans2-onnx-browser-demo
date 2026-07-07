import { LANGUAGES } from './transliterate.js'
import { MODEL_CONFIGS, loadModel, unloadModel, translate } from './translator.js'

// DOM elements
const selectConfig = document.getElementById('model-config')
const selectPrecision = document.getElementById('model-precision')
const selectProvider = document.getElementById('ort-provider')
const btnLoad = document.getElementById('btn-load')
const btnUnload = document.getElementById('btn-unload')

const loadingCard = document.getElementById('loading-card')
const loadingHeader = document.getElementById('loading-header')
const loadingBody = document.getElementById('loading-body')
const loadingNotes = document.getElementById('loading-notes')
const loadingSummaryBadge = document.getElementById('loading-summary-badge')
const chevronIcon = document.getElementById('chevron-icon')
const progressContainer = document.getElementById('progress-container')

const badgeStatus = document.getElementById('badge-status')
const selectSrcLang = document.getElementById('src-lang')
const selectTgtLang = document.getElementById('tgt-lang')
const textareaSrc = document.getElementById('src-text')
const textareaTgt = document.getElementById('tgt-text')
const btnTranslate = document.getElementById('btn-translate')
const translationLoader = document.getElementById('translation-loader')

const metricsContainer = document.getElementById('metrics')
const metricLoad = document.getElementById('metric-load')
const metricSpeed = document.getElementById('metric-speed')
const metricTotal = document.getElementById('metric-total')
const warningBanner = document.getElementById('quantization-warning')

// Toggle collapsible body
loadingHeader.addEventListener('click', () => {
  const isHidden = loadingBody.classList.contains('hidden')
  if (isHidden) {
    loadingBody.classList.remove('hidden')
    chevronIcon.classList.add('rotate-180')
  } else {
    loadingBody.classList.add('hidden')
    chevronIcon.classList.remove('rotate-180')
  }
})

// Configure languages selectors based on selected model
function updateLanguageSelectors() {
  const configKey = selectConfig.value
  const config = MODEL_CONFIGS[configKey]
  const direction = config.direction

  selectSrcLang.innerHTML = ''
  selectTgtLang.innerHTML = ''

  const indicCodes = Object.keys(LANGUAGES).filter(code => code !== 'eng_Latn')

  if (direction === 'en-indic') {
    selectSrcLang.innerHTML = '<option value="eng_Latn">English</option>'
    indicCodes.forEach(code => {
      selectTgtLang.innerHTML += `<option value="${code}">${LANGUAGES[code]}</option>`
    })
  } else if (direction === 'indic-en') {
    indicCodes.forEach(code => {
      selectSrcLang.innerHTML += `<option value="${code}">${LANGUAGES[code]}</option>`
    })
    selectTgtLang.innerHTML = '<option value="eng_Latn">English</option>'
  } else {
    indicCodes.forEach(code => {
      selectSrcLang.innerHTML += `<option value="${code}">${LANGUAGES[code]}</option>`
      selectTgtLang.innerHTML += `<option value="${code}">${LANGUAGES[code]}</option>`
    })
  }
}

const MODEL_SIZES = {
  'en-indic-200m': {
    'q4f16': '~373 MB',
    'int8': '~295 MB',
    'fp16': '~552 MB',
    'fp32': '~1.1 GB'
  },
  'en-indic-1b': {
    'q4f16': '~1.0 GB',
    'int8': '~1.1 GB',
    'fp16': '~2.1 GB',
    'fp32': '~4.2 GB'
  },
  'indic-en-200m': {
    'q4f16': '~285 MB',
    'int8': '~250 MB',
    'fp16': '~464 MB',
    'fp32': '~899 MB'
  },
  'indic-en-1b': {
    'q4f16': '~854 MB',
    'int8': '~1.0 GB',
    'fp16': '~1.9 GB',
    'fp32': '~3.8 GB'
  },
  'indic-indic-320m': {
    'q4f16': '~480 MB',
    'int8': '~358 MB',
    'fp16': '~659 MB',
    'fp32': '~1.2 GB'
  },
  'indic-indic-1b': {
    'q4f16': '~1.2 GB',
    'int8': '~1.2 GB',
    'fp16': '~2.3 GB',
    'fp32': '~4.5 GB'
  }
}

function updateModelSizeDisplay() {
  const config = selectConfig.value
  const precision = selectPrecision.value
  const displayElement = document.getElementById('model-size-display')
  if (!displayElement) return

  const sizeStr = (MODEL_SIZES[config] && MODEL_SIZES[config][precision]) || 'Unknown size'
  displayElement.textContent = sizeStr
}

selectConfig.addEventListener('change', () => {
  updateLanguageSelectors()
  updateModelSizeDisplay()
})
updateLanguageSelectors()

function checkQuantizationWarning() {
  const precision = selectPrecision.value
  const provider = selectProvider.value
  if ((precision === 'int8' || precision === 'q4f16') && provider === 'webgpu') {
    warningBanner.classList.remove('hidden')
  } else {
    warningBanner.classList.add('hidden')
  }
}

function updateProviderDefault() {
  const precision = selectPrecision.value
  if (precision === 'int8' || precision === 'q4f16') {
    selectProvider.value = 'wasm'
  } else {
    selectProvider.value = 'webgpu'
  }
  checkQuantizationWarning()
}

selectPrecision.addEventListener('change', () => {
  updateProviderDefault()
  updateModelSizeDisplay()
})
selectProvider.addEventListener('change', checkQuantizationWarning)
updateProviderDefault()
updateModelSizeDisplay()

// Setup individual UI progress bar
function createProgressBar(id, name) {
  const div = document.createElement('div')
  div.id = `progress-item-${id}`
  div.className = 'space-y-1.5'
  div.innerHTML = `
    <div class="flex justify-between text-xs font-semibold text-zinc-300">
      <span>${name}</span>
      <span id="progress-val-${id}">0%</span>
    </div>
    <div class="w-full bg-zinc-950 h-2 rounded-full overflow-hidden border border-zinc-800">
      <div id="progress-bar-${id}" class="bg-teal-500 h-full w-0 transition-all duration-150"></div>
    </div>
  `
  progressContainer.appendChild(div)
}

function updateProgressBar(id, progress) {
  const bar = document.getElementById(`progress-bar-${id}`)
  const text = document.getElementById(`progress-val-${id}`)
  if (bar && text) {
    bar.style.width = `${progress}%`
    text.innerText = `${progress}%`
  }
}

// Model Load trigger
btnLoad.addEventListener('click', async () => {
  const configKey = selectConfig.value
  const precision = selectPrecision.value
  const provider = selectProvider.value

  // Reset UI
  progressContainer.innerHTML = ''
  loadingCard.classList.remove('hidden')
  loadingBody.classList.remove('hidden')
  chevronIcon.classList.add('rotate-180')
  loadingSummaryBadge.classList.add('hidden')
  btnLoad.disabled = true
  btnLoad.innerHTML = `
    <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
    Loading Model...
  `
  selectConfig.disabled = true
  selectPrecision.disabled = true
  selectProvider.disabled = true

  // Scroll down to the loading card
  setTimeout(() => {
    loadingCard.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, 100)

  loadingNotes.innerText =
    'Downloading ONNX graph protos and weight sidecars (.onnx.data) to browser memory...'

  const startTime = performance.now()

  try {
    await loadModel(configKey, precision, provider, (id, label, percent) => {
      if (!document.getElementById(`progress-item-${id}`)) {
        createProgressBar(id, label)
      }
      updateProgressBar(id, percent)
    })

    const loadTime = ((performance.now() - startTime) / 1000).toFixed(2)

    // Update UI state to Connected
    badgeStatus.innerText = 'Ready'
    badgeStatus.className = 'px-2 py-0.5 text-xs font-semibold bg-emerald-950 text-emerald-400 border border-emerald-900 rounded-full'
    
    selectSrcLang.disabled = false
    selectTgtLang.disabled = false
    textareaSrc.disabled = false
    btnTranslate.disabled = false
    btnUnload.disabled = false
    btnLoad.innerHTML = 'Load Model'

    // Load metric latency
    metricsContainer.classList.remove('hidden')
    metricLoad.innerText = `${loadTime}s`
    metricSpeed.innerText = '-'
    metricTotal.innerText = '-'

    loadingNotes.innerText = 'All models loaded and sessions initialized successfully! (Click header to toggle logs)'
    setTimeout(() => {
      loadingBody.classList.add('hidden')
      chevronIcon.classList.remove('rotate-180')
      loadingSummaryBadge.classList.remove('hidden')
    }, 800)

  } catch (err) {
    console.error('Failed to load sessions:', err)
    loadingNotes.innerText = `Error loading model: ${err.message}`
    btnLoad.disabled = false
    btnLoad.innerHTML = 'Load Model'
    selectConfig.disabled = false
    selectPrecision.disabled = false
    selectProvider.disabled = false
  }
})

// Unload trigger
btnUnload.addEventListener('click', () => {
  unloadModel()

  badgeStatus.innerText = 'Disconnected'
  badgeStatus.className = 'px-2 py-0.5 text-xs font-semibold bg-red-950 text-red-400 border border-red-900 rounded-full'
  
  selectSrcLang.disabled = true
  selectTgtLang.disabled = true
  textareaSrc.disabled = true
  btnTranslate.disabled = true
  btnUnload.disabled = true
  btnLoad.disabled = false
  selectConfig.disabled = false
  selectPrecision.disabled = false
  selectProvider.disabled = false

  loadingCard.classList.add('hidden')
  loadingBody.classList.add('hidden')
  loadingSummaryBadge.classList.add('hidden')
  chevronIcon.classList.remove('rotate-180')
  metricsContainer.classList.add('hidden')
  textareaSrc.value = ''
  textareaTgt.value = ''
})

// Translation Runner
btnTranslate.addEventListener('click', async () => {
  const text = textareaSrc.value.trim()
  if (!text) return

  const srcLang = selectSrcLang.value
  const tgtLang = selectTgtLang.value

  translationLoader.classList.remove('hidden')
  btnTranslate.disabled = true
  btnTranslate.textContent = 'Translating...'

  // Disable controls during translation to gray them out and prevent concurrent modifications
  textareaSrc.disabled = true
  selectSrcLang.disabled = true
  selectTgtLang.disabled = true
  selectConfig.disabled = true
  selectPrecision.disabled = true
  selectProvider.disabled = true
  btnUnload.disabled = true

  // Yield to the browser event loop to guarantee a repaint of the disabled states
  await new Promise(resolve => setTimeout(resolve, 50))

  try {
    const result = await translate(text, srcLang, tgtLang)
    
    textareaTgt.value = result.translation

    // Calculate token/s generation speed ignoring TTFT latency
    const speed = (result.totalTokens / ((result.totalTimeMs - (result.ttftTime || 0)) / 1000)).toFixed(1)
    
    metricSpeed.innerText = `${speed} tok/s`
    metricTotal.innerText = `${(result.totalTimeMs / 1000).toFixed(2)}s`

  } catch (err) {
    console.error('Translation failed:', err)
    textareaTgt.value = `Translation failed: ${err.message}`
  } finally {
    translationLoader.classList.add('hidden')
    btnTranslate.disabled = false
    btnTranslate.textContent = 'Translate'

    // Re-enable controls
    textareaSrc.disabled = false
    selectSrcLang.disabled = false
    selectTgtLang.disabled = false
    selectConfig.disabled = false
    selectPrecision.disabled = false
    selectProvider.disabled = false
    btnUnload.disabled = false
  }
})
