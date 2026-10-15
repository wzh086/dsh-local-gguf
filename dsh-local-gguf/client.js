/**
 * Client half: the local-model configuration panel.
 *
 * It renders inside the Models settings page, in the per-provider extension area
 * (`settings.models.provider-card`), for the two routes this plugin owns. The
 * shape follows LM Studio's model loader deliberately: a model chooser, the load
 * parameters that matter, and a live memory estimate split into VRAM and system
 * RAM, with the reasoning behind the recommended values.
 *
 * The panel is plain JavaScript and React from the browser module table. It
 * imports no Harness Client package — those change without notice and a throwing
 * component blanks its slot entry — and styles only with `--dsw-alias-*` theme
 * tokens, so it reads correctly in both themes.
 */

window.__ModuleLoader__.load({
  id: '@local/dsh-local-gguf',
  factory(require) {
    const React = require('react')
    const h = React.createElement
    const { useCallback, useEffect, useMemo, useRef, useState } = React

    /** Routes this panel configures; anything else in the slot is not ours. */
    const OWNED = ['lmstudio', 'local-gguf']
    const BASE = '/local-gguf'

    /**
     * Localization.
     *
     * Text goes through the Client locale service, so the panel follows whatever
     * language the application is set to rather than the language this file was
     * written in. The dictionaries below are registered for both shipped locales
     * and are also the fallback when the service is absent or does not know the
     * active language — a translated panel is more useful than an English one in a
     * Chinese application, which is the case that matters here.
     */
    const NS = 'local-gguf-panel'
    const DICT = {
      en: {
        title: 'Local GGUF models',
        rescan: 'Rescan',
        scanning: 'Scanning…',
        folders: 'Model folders',
        foldersSaved: 'Saved to this plugin’s configuration',
        foldersReadOnly: 'read-only in this composition',
        remove: 'Remove',
        removed: 'Removed {path}',
        noFolders: 'No extra folder configured; LM Studio’s own folders are still scanned.',
        addFolderPlaceholder: 'D:\\models  — add a folder to scan for .gguf files',
        addFolder: 'Add folder',
        added: 'Added {path}',
        scannedNow: 'Scanned now: {list}',
        none: 'none',
        downloadPlaceholder: 'Download from LM Studio Hub / Hugging Face, e.g. qwen2.5-7b-instruct',
        download: 'Download',
        downloadDone: 'Download finished. Press Rescan.',
        downloadHint: 'Downloads run through LM Studio’s `lms get`, so LM Studio must be running. A downloaded model lands in a folder this plugin already scans.',
        probing: 'probing hardware…',
        serverUp: 'LM Studio server reachable at {url}',
        serverDown: 'LM Studio server not running (started on demand)',
        counts: '{models} chat models · {dirs} folders',
        runtimes: 'runtimes: {list}',
        modelFile: 'Model file',
        selectModel: 'Select a .gguf model…',
        noModels: 'No GGUF files found. Add a folder above, or download a model in LM Studio.',
        contextLength: 'Context length',
        contextHint: 'Model supports up to {tokens} tokens',
        gpuLayers: 'GPU layers',
        gpuLayersHint: '0 – {layers} blocks offloaded to VRAM',
        threads: 'CPU threads',
        batchSize: 'Eval batch size',
        batchSizeHint: 'Logical batch (-b)',
        ubatchSize: 'Physical batch size',
        ubatchSizeHint: 'Physical batch (-ub)',
        kvType: 'KV cache type',
        flashAttention: 'Flash attention',
        offloadKv: 'Offload KV cache to GPU',
        mmap: 'Memory-map the file',
        estimatedMemory: 'Estimated memory usage',
        memorySummary: 'GPU {gpu} · Total {total}',
        vram: 'VRAM',
        systemRam: 'System RAM',
        ofTotal: '{used} of {total}',
        weights: 'weights {value}',
        kvCache: 'KV cache {value}',
        compute: 'compute + overhead {value}',
        measuredSummary: 'llama.cpp measured: GPU {gpu} · RAM {ram} · total {total}',
        measuredBreakdown: 'weights {weights} · KV {kv} · compute {compute}',
        measuredOffload: 'offloaded {layers} layers · n_ctx {ctx}',
        measuredNone: 'llama.cpp reported n_ctx {ctx}; no allocation lines were captured',
        overflow: 'This configuration exceeds the detected memory. Lower the context length, offload fewer layers, or quantize the KV cache.',
        useRecommended: 'Use recommended values',
        saveDefaults: 'Save as defaults',
        saveDefaultsTitle: 'Write these load parameters into this plugin’s configuration so they are used for every future load',
        savedDefaults: 'Saved as this route’s defaults.',
        clearDefaults: 'Clear defaults',
        clearDefaultsTitle: 'Clear the pinned parameters and let the recommendation engine decide again',
        clearedDefaults: 'Cleared; the recommendation is used again.',
        copyYaml: 'Copy as YAML',
        appliedLmStudio: 'Applied when LM Studio loads this model.',
        appliedLocal: 'Applied when this plugin starts llama-server on this file.',
        readingHeader: 'Reading the GGUF header…',
        runningBackends: 'Running llama-server processes started by this plugin:',
        crashed: 'Local GGUF panel failed to render: {message}',
        startup: 'Large models take a while to load; the first reply waits for the load.',
        running: 'running',
        stopped: 'stopped',
        note_noGpu: 'No GPU was detected, so the model runs on the CPU with all layers in system memory.',
        note_cpuThreads: 'Using {threads} generation threads, matching the physical core count.',
        note_vramReserve: 'Reserving {reserve} of {total} VRAM for the desktop and other applications.',
        note_allLayersFit: 'All {layers} layers fit in VRAM, so the whole model is GPU-resident.',
        note_partialOffload: 'Offloading {offloaded} of {layers} layers to the GPU; the rest run on the CPU.',
        note_noFit: 'The weights do not fit in VRAM even with the cache in RAM, so the model runs on the CPU.',
        note_raisedContext: 'Raised the context to {context} tokens, which still fits the VRAM budget.',
        note_quantizedKv: 'Quantized the KV cache to q8_0 to bring the estimate inside the VRAM budget.',
        note_moeAdvice: 'This is a mixture-of-experts model ({used} of {total} experts per token); keeping the expert weights on the CPU with --n-cpu-moe usually beats offloading whole layers.',
        note_weightsNearRam: 'Weights ({weights}) are close to total system memory ({ram}); expect swapping.',
        note_partialOffloadSlow: 'Partial offload splits every token across PCIe; expect lower throughput than a fully GPU-resident model.',
        note_smallerQuant: 'Consider a smaller quantization (for example Q4_K_M) to fit this model in VRAM.',
        note_kvQuality: 'q8_0 KV cache slightly reduces long-context quality; f16 is preferable when memory allows.',
        note_estimatedVram: 'VRAM for "{gpu}" was read from a 32-bit registry value and may understate cards above 4 GB.',
        note_ramOverflow: 'Estimated system memory use ({used}) exceeds 90% of installed RAM ({ram}).',
      },
      zh: {
        title: '本地 GGUF 模型',
        rescan: '重新扫描',
        scanning: '扫描中…',
        folders: '模型目录',
        foldersSaved: '保存到本插件配置',
        foldersReadOnly: '当前组合下只读',
        remove: '移除',
        removed: '已移除 {path}',
        noFolders: '未配置额外目录；LM Studio 自己的模型目录仍会被扫描。',
        addFolderPlaceholder: 'D:\\models —— 填写要扫描 .gguf 的目录',
        addFolder: '添加目录',
        added: '已添加 {path}',
        scannedNow: '本次扫描目录：{list}',
        none: '无',
        downloadPlaceholder: '从 LM Studio Hub / Hugging Face 下载，例如 qwen2.5-7b-instruct',
        download: '下载',
        downloadDone: '下载完成，点「重新扫描」即可看到。',
        downloadHint: '下载通过 LM Studio 的 `lms get` 执行，因此需要 LM Studio 正在运行。下载好的模型会落到本插件已经在扫描的目录里。',
        probing: '正在探测硬件…',
        serverUp: 'LM Studio 服务可访问：{url}',
        serverDown: 'LM Studio 服务未运行（需要时会自动启动）',
        counts: '{models} 个对话模型 · {dirs} 个目录',
        runtimes: '运行时：{list}',
        modelFile: '模型文件',
        selectModel: '选择一个 .gguf 模型…',
        noModels: '没有找到 GGUF 文件。请在上方添加目录，或在 LM Studio 中下载模型。',
        contextLength: '上下文长度',
        contextHint: '模型最多支持 {tokens} token',
        gpuLayers: 'GPU 卸载层数',
        gpuLayersHint: '共 {layers} 层，可卸载到显存',
        threads: 'CPU 线程数',
        batchSize: '评估批处理大小',
        batchSizeHint: '逻辑批大小 (-b)',
        ubatchSize: '物理批处理大小',
        ubatchSizeHint: '物理批大小 (-ub)',
        kvType: 'KV 缓存类型',
        flashAttention: '快速注意力',
        offloadKv: '将 KV 缓存卸载到 GPU',
        mmap: '使用内存映射 (mmap)',
        estimatedMemory: '预估内存占用',
        memorySummary: 'GPU {gpu} · 合计 {total}',
        vram: '显存',
        systemRam: '系统内存',
        ofTotal: '{used} / {total}',
        weights: '权重 {value}',
        kvCache: 'KV 缓存 {value}',
        compute: '计算缓冲与开销 {value}',
        measuredSummary: 'llama.cpp 实测：GPU {gpu} · 内存 {ram} · 合计 {total}',
        measuredBreakdown: '权重 {weights} · KV 缓存 {kv} · 计算缓冲 {compute}',
        measuredOffload: '已卸载 {layers} 层 · n_ctx {ctx}',
        measuredNone: 'llama.cpp 报告了 n_ctx {ctx}，但未捕获到内存分配行',
        overflow: '该配置超出了检测到的可用内存。请降低上下文长度、减少卸载层数，或量化 KV 缓存。',
        useRecommended: '使用推荐值',
        saveDefaults: '存为默认值',
        saveDefaultsTitle: '把这些加载参数写入本插件配置，之后每次加载都会使用',
        savedDefaults: '已保存为该线路的默认值。',
        clearDefaults: '清除默认值',
        clearDefaultsTitle: '清除固定参数，交回推荐引擎决定',
        clearedDefaults: '已清除，重新使用推荐值。',
        copyYaml: '复制为 YAML',
        appliedLmStudio: '在 LM Studio 加载该模型时生效。',
        appliedLocal: '在本插件为该文件启动 llama-server 时生效。',
        readingHeader: '正在读取 GGUF 头部…',
        runningBackends: '本插件启动中的 llama-server 进程：',
        crashed: '本地 GGUF 面板渲染失败：{message}',
        startup: '大模型加载需要一些时间，首次回复会等待加载完成。',
        running: '运行中',
        stopped: '已停止',
        note_noGpu: '未检测到 GPU，模型将全部在内存中由 CPU 运行。',
        note_cpuThreads: '使用 {threads} 个生成线程，与物理核心数一致。',
        note_vramReserve: '从 {total} 显存中预留 {reserve} 给桌面和其他应用。',
        note_allLayersFit: '全部 {layers} 层都能放进显存，整个模型常驻 GPU。',
        note_partialOffload: '将 {layers} 层中的 {offloaded} 层卸载到 GPU，其余由 CPU 运行。',
        note_noFit: '即使 KV 缓存放在内存里，权重也放不进显存，因此模型由 CPU 运行。',
        note_raisedContext: '在显存预算内把上下文提高到 {context} token。',
        note_quantizedKv: '把 KV 缓存量化为 q8_0，使预估占用落回显存预算内。',
        note_moeAdvice: '这是 MoE（混合专家）模型，每个 token 使用 {total} 个专家中的 {used} 个；用 --n-cpu-moe 把专家权重留在 CPU 通常比整层卸载更快。',
        note_weightsNearRam: '权重（{weights}）已接近系统内存总量（{ram}），可能会出现换页。',
        note_partialOffloadSlow: '部分卸载会让每个 token 都经过 PCIe，吞吐会低于全部常驻 GPU 的情况。',
        note_smallerQuant: '可以考虑更小的量化（例如 Q4_K_M）以便装进显存。',
        note_kvQuality: 'q8_0 的 KV 缓存会略微影响长上下文质量；显存够时 f16 更好。',
        note_estimatedVram: '「{gpu}」的显存来自 32 位注册表值，超过 4 GB 的显卡可能被低估。',
        note_ramOverflow: '预估内存占用（{used}）超过已安装内存（{ram}）的 90%。',
      },
    }

    /** The locale service, once `apply` has seen it. */
    let localeService
    /** Bound translate function for this namespace. */
    let boundTranslate

    /** Pick the dictionary to use when the service cannot answer. */
    function fallbackLanguage() {
      if (localeService?.getLocale) {
        const active = String(localeService.getLocale()?.active ?? '')
        if (active.toLowerCase().startsWith('zh')) return 'zh'
        if (active) return 'en'
      }
      const navigatorLanguage = typeof navigator !== 'undefined' ? String(navigator.language ?? '') : ''
      const documentLanguage = typeof document !== 'undefined' ? String(document.documentElement?.lang ?? '') : ''
      return `${navigatorLanguage} ${documentLanguage}`.toLowerCase().includes('zh') ? 'zh' : 'en'
    }

    /**
     * Translate one key, substituting `{name}` placeholders.
     *
     * The locale service is asked first; when it has no registration for the active
     * language it answers with the key itself, which is the signal to fall back to
     * the built-in dictionary rather than rendering a raw key.
     */
    function t(key, params) {
      let text
      if (boundTranslate) {
        const translated = boundTranslate(key)
        if (typeof translated === 'string' && translated !== key) text = translated
      }
      if (text === undefined) text = DICT[fallbackLanguage()]?.[key] ?? DICT.en[key] ?? key
      if (!params) return text
      return text.replace(/\{(\w+)\}/g, (match, name) => (name in params ? String(params[name]) : match))
    }

    const T = {
      bg: 'var(--dsw-alias-bg-layer-1)',
      bgNested: 'var(--dsw-alias-bg-layer-2)',
      border: 'var(--dsw-alias-border-l1)',
      borderStrong: 'var(--dsw-alias-border-l2)',
      text: 'var(--dsw-alias-label-primary)',
      dim: 'var(--dsw-alias-label-secondary)',
      brand: 'var(--dsw-alias-brand-primary)',
      warn: 'var(--dsw-alias-state-warn-primary)',
      error: 'var(--dsw-alias-state-error-primary)',
      ok: 'var(--dsw-alias-state-success-primary)',
      idle: 'var(--dsw-alias-state-idle-primary)',
    }

    const label = { fontSize: 12, color: T.dim, lineHeight: '18px' }
    const value = { fontSize: 12, color: T.text, fontVariantNumeric: 'tabular-nums' }
    const row = { display: 'flex', alignItems: 'center', gap: 8, minHeight: 28 }
    const grow = { flex: 1, minWidth: 0 }
    const input = {
      width: 96,
      boxSizing: 'border-box',
      padding: '4px 8px',
      fontSize: 12,
      color: T.text,
      background: T.bgNested,
      border: `1px solid ${T.border}`,
      borderRadius: 6,
      outline: 'none',
      fontVariantNumeric: 'tabular-nums',
    }
    const select = { ...input, width: 116 }
    const button = {
      padding: '4px 10px',
      fontSize: 12,
      color: T.text,
      background: T.bgNested,
      border: `1px solid ${T.border}`,
      borderRadius: 6,
      cursor: 'pointer',
    }
    const buttonPrimary = { ...button, color: '#fff', background: T.brand, borderColor: T.brand }
    const card = { border: `1px solid ${T.border}`, borderRadius: 8, padding: 12, background: T.bg, display: 'grid', gap: 10 }
    const code = {
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
      fontSize: 11,
      color: T.dim,
      background: T.bgNested,
      border: `1px solid ${T.border}`,
      borderRadius: 6,
      padding: '6px 8px',
      whiteSpace: 'pre-wrap',
      wordBreak: 'break-all',
    }

    /** `fetch` with a JSON body and a readable error. */
    async function getJson(path) {
      const response = await fetch(`${BASE}${path}`, { headers: { accept: 'application/json' } })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body?.error ?? `HTTP ${response.status}`)
      return body
    }

    /** POST one write action and surface the Host's diagnostic verbatim. */
    async function postJson(path, payload) {
      const response = await fetch(`${BASE}${path}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', accept: 'application/json' },
        body: JSON.stringify(payload ?? {}),
      })
      const body = await response.json().catch(() => ({}))
      // A write can fail for a specific reason — no such directory, no config
      // editor, LM Studio not running — so that reason is shown, not the status.
      if (!response.ok) throw new Error(body?.error ?? body?.hint ?? `HTTP ${response.status}`)
      return body
    }

    function bytes(n) {
      if (!Number.isFinite(n) || n <= 0) return '0'
      const gib = n / 1024 ** 3
      if (gib >= 1) return `${gib.toFixed(2)} GB`
      return `${(n / 1024 ** 2).toFixed(0)} MB`
    }

    /** Render one rationale/warning note, preferring its localized form. */
    function noteText(note) {
      if (note?.key) {
        const localized = DICT[fallbackLanguage()]?.[`note_${note.key}`] ?? DICT.en[`note_${note.key}`]
        if (localized) return t(`note_${note.key}`, note.params ?? {})
      }
      // A key this client build does not know falls back to the Host's prose,
      // which is always present and always truthful.
      return String(note?.text ?? '')
    }

    /** A labelled track+fill bar, the shape LM Studio uses for memory. */
    function Bar(props) {
      const pct = Math.max(0, Math.min(100, props.percent))
      return h(
        'div',
        { style: { display: 'grid', gap: 4 } },
        h(
          'div',
          { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' } },
          h('span', { style: label }, props.title),
          h('span', { style: { ...value, color: props.overflow ? T.error : T.text } }, props.caption),
        ),
        h(
          'div',
          { style: { height: 6, borderRadius: 3, background: T.bgNested, border: `1px solid ${T.border}`, overflow: 'hidden' } },
          h('div', {
            style: {
              width: `${pct}%`,
              height: '100%',
              background: props.overflow ? T.error : props.color ?? T.brand,
              transition: 'width 120ms ease',
            },
          }),
        ),
      )
    }

    function NumberField(props) {
      return h(
        'div',
        { style: { ...row, justifyContent: 'space-between' } },
        h('span', { style: label, title: props.hint }, props.label),
        h('input', {
          style: input,
          type: 'number',
          min: props.min,
          max: props.max,
          value: props.value ?? '',
          placeholder: props.placeholder,
          onChange: (event) => {
            const raw = event.target.value
            props.onChange(raw === '' ? undefined : Number(raw))
          },
        }),
      )
    }

    function Panel(props) {
      // The panel renders in two places: inside a provider card (where `provider`
      // identifies whose card it is) and in the Models page footer, which supplies
      // no provider and always renders. In a card it must stay silent unless the
      // card belongs to one of this plugin's routes.
      const providerId = props?.provider?.provider
      const inFooter = props?.provider === undefined
      const [status, setStatus] = useState(null)
      const [models, setModels] = useState(null)
      const [selected, setSelected] = useState(null)
      const [detail, setDetail] = useState(null)
      const [config, setConfig] = useState(null)
      const [busy, setBusy] = useState(false)
      const [error, setError] = useState(null)
      const [notice, setNotice] = useState(null)
      const [newDir, setNewDir] = useState('')
      const [downloadQuery, setDownloadQuery] = useState('')
      const [writeBusy, setWriteBusy] = useState(false)
      const mounted = useRef(true)

      useEffect(() => {
        mounted.current = true
        return () => {
          mounted.current = false
        }
      }, [])

      const loadStatus = useCallback(async (refresh) => {
        try {
          const [nextStatus, nextModels] = await Promise.all([
            getJson(`/status${refresh ? '?refresh=1' : ''}`),
            getJson(`/models${refresh ? '?refresh=1' : ''}`),
          ])
          if (!mounted.current) return
          setStatus(nextStatus)
          setModels(nextModels)
          setError(null)
        } catch (cause) {
          if (mounted.current) setError(cause.message)
        }
      }, [])

      useEffect(() => {
        if (!inFooter && !OWNED.includes(providerId)) return
        loadStatus(false)
      }, [inFooter, providerId, loadStatus])

      // Fetch the full inspection whenever the chosen model or a parameter changes,
      // so every displayed number describes exactly the controls on screen.
      useEffect(() => {
        if (!selected) return
        let cancelled = false
        const query = new URLSearchParams({ model: selected })
        if (config) {
          query.set('context_length', String(config.contextLength))
          query.set('gpu_layers', String(config.gpuLayers))
          query.set('kv_type', String(config.kvTypeK))
          query.set('batch_size', String(config.batchSize))
          query.set('ubatch_size', String(config.ubatchSize))
          query.set('threads', String(config.threads))
          query.set('flash_attention', config.flashAttention ? '1' : '0')
          query.set('offload_kv', config.offloadKvToGpu ? '1' : '0')
        }
        ;(async () => {
          try {
            const next = await getJson(`/inspect?${query.toString()}`)
            if (cancelled || !mounted.current) return
            setDetail(next)
            setConfig((current) => current ?? next.config)
            setError(null)
          } catch (cause) {
            if (!cancelled && mounted.current) setError(cause.message)
          }
        })()
        return () => {
          cancelled = true
        }
        // `config` is intentionally a dependency: each edit re-estimates.
      }, [selected, config])

      if (!inFooter && !OWNED.includes(providerId)) return null

      const isLmStudio = providerId === 'lmstudio'
      const usableVram = status?.hardware?.usableVram ?? 0
      const totalRam = status?.hardware?.memory?.total ?? 0
      const gpu = status?.hardware?.gpus?.find((entry) => !entry.integrated)
      const estimate = detail?.estimate
      const overflowsVram = estimate && usableVram > 0 && estimate.gpuBytes > usableVram
      const ramLimit = totalRam * 0.9
      const overflowsRam = estimate && totalRam > 0 && estimate.ramBytes > ramLimit

      const chatModels = useMemo(
        () => (models?.models ?? []).filter((model) => model.kind === 'llm'),
        [models],
      )

      const set = (patch) => setConfig((current) => ({ ...(current ?? {}), ...patch }))

      /** Run one write action, then refresh and report the outcome. */
      const write = async (path, payload, successMessage) => {
        setWriteBusy(true)
        setNotice(null)
        setError(null)
        try {
          const result = await postJson(path, payload)
          setNotice(typeof successMessage === 'function' ? successMessage(result) : successMessage)
          await loadStatus(true)
        } catch (cause) {
          setError(cause.message)
        } finally {
          setWriteBusy(false)
        }
      }

      const configuredDirectories = models?.configuredDirectories ?? []
      const discoveredDirectories = models?.directories ?? []
      const canWrite = models?.canWrite !== false

      return h(
        'div',
        { style: { ...card, marginTop: 8 } },
        // ---- header -------------------------------------------------------
        h(
          'div',
          { style: { ...row, flexWrap: 'wrap' } },
          h('strong', { style: { fontSize: 13, color: T.text } }, t('title')),
          h(
            'span',
            { style: { ...label, marginLeft: 'auto' } },
            status
              ? `${gpu ? `${gpu.name} · ${bytes(gpu.vramBytes)} ${t('vram')}` : t('none')} · ${bytes(totalRam)} ${t('systemRam')}`
              : t('probing'),
          ),
          h(
            'button',
            { style: button, onClick: () => { setBusy(true); loadStatus(true).finally(() => setBusy(false)) }, disabled: busy },
            busy ? t('scanning') : t('rescan'),
          ),
        ),

        error ? h('div', { style: { ...label, color: T.error } }, error) : null,

        // ---- model directories and downloads -----------------------------
        h('div', { style: { height: 1, background: T.border } }),
        h(
          'div',
          { style: { display: 'grid', gap: 6 } },
          h(
            'div',
            { style: row },
            h('strong', { style: { fontSize: 12, color: T.text } }, t('folders')),
            h('span', { style: { ...label, marginLeft: 'auto' } },
              canWrite ? t('foldersSaved') : t('foldersReadOnly')),
          ),
          configuredDirectories.length
            ? configuredDirectories.map((dir) =>
                h(
                  'div',
                  { key: dir, style: { ...row, gap: 8 } },
                  h('span', { style: { ...label, flex: 1, wordBreak: 'break-all', color: T.text } }, dir),
                  h('button', {
                    style: button,
                    disabled: writeBusy || !canWrite,
                    onClick: () => write('/directories', { action: 'remove', path: dir }, t('removed', { path: dir })),
                  }, t('remove')),
                ),
              )
            : h('span', { style: label }, t('noFolders')),
          h(
            'div',
            { style: row },
            h('input', {
              style: { ...input, width: 'auto', flex: 1, minWidth: 0 },
              placeholder: t('addFolderPlaceholder'),
              value: newDir,
              onChange: (event) => setNewDir(event.target.value),
              onKeyDown: (event) => {
                if (event.key !== 'Enter' || !newDir.trim() || writeBusy || !canWrite) return
                write('/directories', { action: 'add', path: newDir.trim() }, t('added', { path: newDir.trim() }))
                setNewDir('')
              },
            }),
            h('button', {
              style: buttonPrimary,
              disabled: writeBusy || !canWrite || !newDir.trim(),
              onClick: () => {
                write('/directories', { action: 'add', path: newDir.trim() }, t('added', { path: newDir.trim() }))
                setNewDir('')
              },
            }, t('addFolder')),
          ),
          h('span', { style: label }, t('scannedNow', { list: discoveredDirectories.join('  ·  ') || t('none') })),

          h(
            'div',
            { style: row },
            h('input', {
              style: { ...input, width: 'auto', flex: 1, minWidth: 0 },
              placeholder: t('downloadPlaceholder'),
              value: downloadQuery,
              onChange: (event) => setDownloadQuery(event.target.value),
            }),
            h('button', {
              style: button,
              disabled: writeBusy || !downloadQuery.trim(),
              onClick: () => write('/download', { query: downloadQuery.trim() }, (r) => r.output || t('downloadDone')),
            }, t('download')),
          ),
          h('span', { style: label }, t('downloadHint')),
        ),

        notice ? h('div', { style: { ...label, color: T.text, whiteSpace: 'pre-wrap', maxHeight: 160, overflow: 'auto' } }, notice) : null,

        // ---- connection state --------------------------------------------
        h(
          'div',
          { style: { ...row, gap: 16, flexWrap: 'wrap' } },
          h('span', { style: { ...label, color: status?.server?.reachable ? T.ok : T.idle } },
            status?.server?.reachable
              ? t('serverUp', { url: status.server.baseUrl })
              : t('serverDown')),
          h('span', { style: label }, t('counts', { models: chatModels.length, dirs: models?.directories?.length ?? 0 })),
          status?.lmStudio?.backends?.length
            ? h('span', { style: label }, t('runtimes', { list: status.lmStudio.backends.slice(0, 2).map((b) => b.acceleration).join(', ') }))
            : null,
        ),

        // ---- model chooser -----------------------------------------------
        h(
          'div',
          { style: row },
          h('span', { style: { ...label, minWidth: 84 } }, t('modelFile')),
          h(
            'select',
            {
              style: { ...select, width: 'auto', flex: 1, minWidth: 0 },
              value: selected ?? '',
              onChange: (event) => {
                const next = event.target.value || null
                setSelected(next)
                setDetail(null)
                setConfig(null)
              },
            },
            h('option', { value: '' }, t('selectModel')),
            chatModels.map((model) =>
              h('option', { key: model.id, value: model.id },
                `${model.name} · ${model.quantization} · ${model.fileSizeText} · ${model.arch}`),
            ),
          ),
        ),

        !chatModels.length
          ? h('div', { style: label }, t('noModels'))
          : null,

        // ---- load parameters ---------------------------------------------
        detail && config
          ? h(
              React.Fragment,
              null,
              h('div', { style: { height: 1, background: T.border } }),

              h(
                'div',
                { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '4px 24px' } },

                h(NumberField, {
                  label: t('contextLength'),
                  hint: t('contextHint', { tokens: detail.model.contextLength?.toLocaleString?.() ?? '?' }),
                  value: config.contextLength,
                  min: 256,
                  max: detail.model.contextLength ?? 1048576,
                  onChange: (v) => set({ contextLength: v ?? 4096 }),
                }),
                h(NumberField, {
                  label: t('gpuLayers'),
                  hint: t('gpuLayersHint', { layers: detail.model.blockCount }),
                  value: config.gpuLayers,
                  min: 0,
                  max: detail.model.blockCount ?? 0,
                  onChange: (v) => set({ gpuLayers: v ?? 0 }),
                }),
                h(NumberField, {
                  label: t('threads'),
                  value: config.threads,
                  min: 1,
                  max: 256,
                  onChange: (v) => set({ threads: v ?? 1 }),
                }),
                h(NumberField, {
                  label: t('batchSize'),
                  hint: t('batchSizeHint'),
                  value: config.batchSize,
                  min: 32,
                  max: 8192,
                  onChange: (v) => set({ batchSize: v ?? 512 }),
                }),
                h(NumberField, {
                  label: t('ubatchSize'),
                  hint: t('ubatchSizeHint'),
                  value: config.ubatchSize,
                  min: 16,
                  max: 2048,
                  onChange: (v) => set({ ubatchSize: v ?? 256 }),
                }),

                h(
                  'div',
                  { style: { ...row, justifyContent: 'space-between' } },
                  h('span', { style: label }, t('kvType')),
                  h(
                    'select',
                    { style: select, value: config.kvTypeK, onChange: (e) => set({ kvTypeK: e.target.value, kvTypeV: e.target.value }) },
                    (detail.presets?.kvCacheTypes ?? ['f16', 'q8_0', 'q4_0']).map((type) => h('option', { key: type, value: type }, type)),
                  ),
                ),
                h(Toggle, { title: t('flashAttention'), checked: config.flashAttention, onChange: (v) => set({ flashAttention: v }) }),
                h(Toggle, { title: t('offloadKv'), checked: config.offloadKvToGpu, onChange: (v) => set({ offloadKvToGpu: v }) }),
                h(Toggle, { title: t('mmap'), checked: config.mmap !== false, onChange: (v) => set({ mmap: v }) }),
              ),

              // ---- memory estimate -----------------------------------------
              h(
                'div',
                { style: { ...card, background: T.bgNested, gap: 8 } },
                h(
                  'div',
                  { style: row },
                  h('strong', { style: { fontSize: 12, color: T.text } }, t('estimatedMemory')),
                  h('span', { style: { ...label, marginLeft: 'auto' } },
                    t('memorySummary', { gpu: estimate.gpu, total: estimate.total })),
                ),
                h(Bar, {
                  title: t('vram'),
                  caption: t('ofTotal', { used: estimate.gpu, total: bytes(usableVram) }),
                  percent: usableVram ? (estimate.gpuBytes / usableVram) * 100 : 0,
                  overflow: overflowsVram,
                  color: T.brand,
                }),
                h(Bar, {
                  title: t('systemRam'),
                  caption: t('ofTotal', { used: estimate.ram, total: bytes(totalRam) }),
                  percent: totalRam ? (estimate.ramBytes / totalRam) * 100 : 0,
                  overflow: overflowsRam,
                  color: T.ok,
                }),
                h('div', { style: { ...label, display: 'flex', gap: 16, flexWrap: 'wrap' } },
                  h('span', null, t('weights', { value: estimate.weightsText })),
                  h('span', null, t('kvCache', { value: estimate.kvCacheText })),
                  h('span', null, t('compute', { value: estimate.computeText }))),
                detail.measured && detail.measured.totalBytes
                  ? h(
                      'div',
                      { style: { ...label, display: 'grid', gap: 2 } },
                      h('span', { style: { color: T.ok } },
                        t('measuredSummary', {
                          gpu: bytes(detail.measured.gpuBytes),
                          ram: bytes(detail.measured.cpuBytes),
                          total: bytes(detail.measured.totalBytes),
                        })),
                      h('span', null,
                        t('measuredBreakdown', {
                          weights: `${bytes(detail.measured.weightsGpuBytes)} GPU + ${bytes(detail.measured.weightsCpuBytes)} RAM`,
                          kv: `${bytes(detail.measured.kvGpuBytes)} GPU + ${bytes(detail.measured.kvCpuBytes)} RAM`,
                          compute: `${bytes(detail.measured.computeGpuBytes)} GPU + ${bytes(detail.measured.computeCpuBytes)} RAM`,
                        })),
                      detail.measured.offloadedLayers
                        ? h('span', null, t('measuredOffload', { layers: detail.measured.offloadedLayers, ctx: detail.measured.contextSize ?? '?' }))
                        : null,
                    )
                  : detail.measured && detail.measured.contextSize
                    ? h('div', { style: label }, t('measuredNone', { ctx: detail.measured.contextSize }))
                    : null,
              ),

              // ---- reasoning -----------------------------------------------
              // The Host sends each line as both prose and a key with parameters:
              // the key is rendered here in the active language, and the prose is
              // the fallback when a key is unknown to this client build.
              detail.rationaleNotes?.length || detail.warningNotes?.length ||
              detail.rationale?.length || detail.warnings?.length
                ? h(
                    'div',
                    { style: { display: 'grid', gap: 2 } },
                    (detail.rationaleNotes?.length ? detail.rationaleNotes : (detail.rationale ?? []).map((text) => ({ key: null, text })))
                      .map((note, index) => h('div', { key: `r${index}`, style: label }, `· ${noteText(note)}`)),
                    (detail.warningNotes?.length ? detail.warningNotes : (detail.warnings ?? []).map((text) => ({ key: null, text })))
                      .map((note, index) =>
                        h('div', { key: `w${index}`, style: { ...label, color: T.warn } }, `! ${noteText(note)}`)),
                  )
                : null,

              overflowsVram || overflowsRam
                ? h('div', { style: { ...label, color: T.error } }, t('overflow'))
                : null,

              // ---- actions --------------------------------------------------
              h(
                'div',
                { style: { ...row, gap: 8, flexWrap: 'wrap' } },
                h('button', {
                  style: button,
                  onClick: () => setConfig({ ...detail.recommendedConfig, threads: detail.recommendedConfig.threads }),
                }, t('useRecommended')),
                h('button', {
                  style: buttonPrimary,
                  disabled: writeBusy || !canWrite,
                  title: t('saveDefaultsTitle'),
                  onClick: () =>
                    write(
                      '/options',
                      {
                        provider: providerId,
                        values: {
                          contextLength: config.contextLength,
                          gpuLayers: config.gpuLayers,
                          batchSize: config.batchSize,
                          ubatchSize: config.ubatchSize,
                          threads: config.threads,
                          kvTypeK: config.kvTypeK,
                          kvTypeV: config.kvTypeV,
                          flashAttention: config.flashAttention,
                          offloadKvToGpu: config.offloadKvToGpu,
                        },
                      },
                      t('savedDefaults'),
                    ),
                }, t('saveDefaults')),
                h('button', {
                  style: button,
                  disabled: writeBusy || !canWrite,
                  title: t('clearDefaultsTitle'),
                  onClick: () =>
                    write(
                      '/options',
                      {
                        provider: providerId,
                        values: {
                          contextLength: null,
                          gpuLayers: null,
                          batchSize: null,
                          ubatchSize: null,
                          threads: null,
                          kvTypeK: null,
                          kvTypeV: null,
                          flashAttention: null,
                          offloadKvToGpu: null,
                        },
                      },
                      t('clearedDefaults'),
                    ),
                }, t('clearDefaults')),
                h('button', {
                  style: button,
                  onClick: () => {
                    const text = `- id: local-gguf\n  config:\n    providers:\n      ${providerId}:\n        contextLength: ${config.contextLength}\n        gpuLayers: ${config.gpuLayers}\n`
                    if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text)
                  },
                }, t('copyYaml')),
                h('span', { style: { ...label, marginLeft: 'auto' } },
                  isLmStudio ? t('appliedLmStudio') : t('appliedLocal')),
              ),

              h('div', { style: code }, `llama-server ${(detail.llamaServerArgs ?? []).join(' ')}`),
            )
          : selected
            ? h('div', { style: label }, t('readingHeader'))
            : null,

        // ---- managed backends ------------------------------------------
        status?.managedBackends?.length
          ? h(
              'div',
              { style: { display: 'grid', gap: 2 } },
              h('span', { style: label }, t('runningBackends')),
              status.managedBackends.map((backend) =>
                h('div', { key: backend.modelPath, style: label },
                  `${backend.modelPath.split(/[\\/]/).pop()} · pid ${backend.pid} · ${backend.baseUrl} · ${backend.running ? t('running') : t('stopped')}`)),
            )
          : null,
      )
    }

    /** A checkbox drawn as a switch, matching the host's control shape. */
    function Toggle(props) {
      return h(
        'div',
        { style: { ...row, justifyContent: 'space-between' } },
        h('span', { style: label }, props.title),
        h(
          'button',
          {
            type: 'button',
            role: 'switch',
            'aria-checked': props.checked ? 'true' : 'false',
            'aria-label': props.title,
            onClick: () => props.onChange(!props.checked),
            style: {
              width: 34,
              height: 18,
              padding: 0,
              borderRadius: 9,
              border: `1px solid ${props.checked ? T.brand : T.border}`,
              background: props.checked ? T.brand : T.bgNested,
              cursor: 'pointer',
              position: 'relative',
              flex: 'none',
            },
          },
          h('span', {
            style: {
              position: 'absolute',
              top: 1,
              left: props.checked ? 17 : 1,
              width: 14,
              height: 14,
              borderRadius: 7,
              background: '#fff',
              transition: 'left 120ms ease',
            },
          }),
        ),
      )
    }

    /**
     * Contain any render failure inside this entry.
     *
     * A throwing component blanks its slot entry, which would take the surrounding
     * Models page down with it, so the failure is caught and shown as one line.
     */
    class Boundary extends React.Component {
      constructor(propsValue) {
        super(propsValue)
        this.state = { failure: null }
      }

      static getDerivedStateFromError(error) {
        return { failure: error }
      }

      render() {
        if (this.state.failure) {
          return h('div', { style: { ...label, color: T.error, marginTop: 8 } },
            t('crashed', { message: this.state.failure.message }))
        }
        return this.props.children
      }
    }

    /**
     * Force a re-render when the active language changes.
     *
     * `t` reads the active locale at call time, so the panel only needs to be told
     * that something changed. Subscribing here rather than threading a locale prop
     * keeps the panel working when it is mounted by a slot owner that knows nothing
     * about this plugin.
     */
    function useLocaleRevision() {
      const [revision, setRevision] = useState(0)
      useEffect(() => {
        if (!localeService?.subscribe) return undefined
        return localeService.subscribe(() => setRevision((value) => value + 1))
      }, [])
      return revision
    }

    function Guarded(props) {
      useLocaleRevision()
      return h(Boundary, null, h(Panel, props))
    }

    return {
      inject: ['slots', 'locale'],
      apply(ctx) {
        // Register the dictionaries so the panel follows the application language,
        // and keep the service so the component can re-render on a switch. Both are
        // optional: without the service the built-in dictionary still follows the
        // browser's language.
        const locale = ctx.get('locale')
        if (locale?.register && locale?.bind) {
          localeService = locale
          ctx.effect(() => locale.register(NS, 'en', DICT.en))
          ctx.effect(() => locale.register(NS, 'zh', DICT.zh))
          boundTranslate = locale.bind(NS)
        }
        // Two seats, deliberately. The footer is an unconditional list slot, so
        // the panel is reachable on the Models page whether or not a provider row
        // renders for this plugin's routes; the provider card is where it belongs
        // when such rows do exist.
        ctx.slots.inject('settings.models.footer', () =>
          ctx.slots.register({ name: 'settings.models.footer', id: 'local-gguf-panel', order: 20 }, Guarded),
        )
        ctx.slots.inject('settings.models.provider-card', () =>
          ctx.slots.register({ name: 'settings.models.provider-card', key: 'local-gguf' }, Guarded),
        )
      },
    }
  },
})
