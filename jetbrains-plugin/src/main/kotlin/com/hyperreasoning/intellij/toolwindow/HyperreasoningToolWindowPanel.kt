package com.hyperreasoning.intellij.toolwindow

import com.hyperreasoning.intellij.backend.BackendStatusService
import com.hyperreasoning.intellij.backend.ClientContext
import com.hyperreasoning.intellij.backend.JobProgressSnapshot
import com.hyperreasoning.intellij.backend.PluginSettingsService
import com.hyperreasoning.intellij.backend.CompareStrategiesRequest
import com.hyperreasoning.intellij.backend.RunHistoryItem
import com.hyperreasoning.intellij.backend.RunLoadResponse
import com.hyperreasoning.intellij.backend.TaskRunRequest
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.fileEditor.FileDocumentManager
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.fileEditor.FileEditorManagerEvent
import com.intellij.openapi.fileEditor.FileEditorManagerListener
import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.openapi.vfs.LocalFileSystem
import com.intellij.openapi.ui.Messages
import com.intellij.ui.JBColor
import com.intellij.ui.components.JBCheckBox
import com.intellij.ui.components.JBLabel
import com.intellij.ui.components.JBPanel
import com.intellij.ui.components.JBScrollPane
import com.intellij.ui.components.JBTextArea
import com.intellij.util.ui.JBUI
import java.awt.BasicStroke
import java.awt.BorderLayout
import java.awt.CardLayout
import java.awt.Color
import java.awt.Cursor
import java.awt.Dimension
import java.awt.Font
import java.awt.GradientPaint
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.GridBagConstraints
import java.awt.GridBagLayout
import java.awt.GridLayout
import java.awt.IllegalComponentStateException
import java.awt.Rectangle
import java.awt.RenderingHints
import java.awt.AlphaComposite
import java.nio.charset.MalformedInputException
import java.nio.file.Files
import java.nio.file.Path
import java.lang.ref.WeakReference
import java.security.MessageDigest
import java.awt.image.BufferedImage
import javax.swing.JButton
import javax.swing.BoxLayout
import javax.swing.JComponent
import javax.swing.JProgressBar
import javax.swing.JPanel
import javax.swing.Scrollable
import javax.swing.SwingConstants
import javax.swing.Timer
import javax.swing.JToggleButton
import javax.swing.UIManager
import javax.swing.event.DocumentEvent
import javax.swing.event.DocumentListener
import kotlin.math.max
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonPrimitive

/**
 * Placeholder UI panel for future task-time search visualization.
 *
 * Planned sections:
 * - task input and run button
 * - branch tree / candidate plans
 * - verifier outcome summary
 * - final patch preview
 * - optional Rainbow vs heuristic vs 1-shot LLM comparison on the current task
 */
class HyperreasoningToolWindowPanel(project: Project) : JBPanel<JBPanel<*>>(BorderLayout()) {
    private val defaultPromptText = "Describe the task to solve..."
    private val statusService = project.getService(BackendStatusService::class.java)
    private val settings = ApplicationManager.getApplication().getService(PluginSettingsService::class.java)
    private val promptArea = JBTextArea(defaultPromptText, 6, 60)
    private val resultsContent = ViewportWidthPanel().apply {
        layout = BoxLayout(this, BoxLayout.Y_AXIS)
        isOpaque = true
        background = Palette.canvas
        border = JBUI.Borders.empty(2)
    }
    private val graphPanel = SearchGraphPanel(project)
    private val runTests = JBCheckBox("Run visible tests", settings.state.runTests)
    private val runHiddenTests = JBCheckBox("Run hidden tests", settings.state.runHiddenTests)
    private val healthButton = ActionButton("Check Backend", ButtonTone.GHOST)
    private val runRandomButton = ActionButton("Random", ButtonTone.SECONDARY)
    private val runRainbowButton = ActionButton("Rainbow", ButtonTone.PRIMARY)
    private val runHeuristicButton = ActionButton("Heuristic", ButtonTone.SECONDARY)
    private val runOneShotButton = ActionButton("1-Shot LLM", ButtonTone.SECONDARY)
    private val compareButton = ActionButton("Compare", ButtonTone.SECONDARY)
    private val refreshHistoryButton = ActionButton("Past Runs", ButtonTone.SECONDARY)
    private val progressBar = RoundedProgressBar().apply {
        isStringPainted = true
        minimum = 0
        maximum = 100
        value = 0
        string = "Idle"
        isVisible = false
    }
    private val progressLabel = JBLabel("Ready")
    private val runStateChip = PulseInfoChip("Run state", progressLabel)
    private val contextArea = JBTextArea("Context\nno active file")
    private val json = Json { ignoreUnknownKeys = true }
    private val rootViewLayout = CardLayout()
    private val rootViewCards = JPanel(rootViewLayout).apply {
        isOpaque = false
    }
    private val historyViewContent = ViewportWidthPanel().apply {
        layout = BoxLayout(this, BoxLayout.Y_AXIS)
        background = Palette.canvas
        border = JBUI.Borders.empty(2, 2, 20, 2)
    }
    private var autoFilledPrompt: String? = null
    private var isUpdatingPromptProgrammatically = false
    private var lastProgressVisualSignature: String? = null

    init {
        background = Palette.canvas
        border = JBUI.Borders.empty(14)
        val content = ViewportWidthPanel().apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            background = Palette.canvas
            border = JBUI.Borders.empty(2, 2, 20, 2)
        }

        configureTextArea(promptArea, false)
        configureContextArea(contextArea)
        configureCheckBox(runTests)
        configureCheckBox(runHiddenTests)
        progressLabel.foreground = Palette.textSecondary
        progressLabel.font = dankMonoFont(Font.PLAIN, 12f)

        promptArea.document.addDocumentListener(object : DocumentListener {
            private fun onChanged() {
                if (!isUpdatingPromptProgrammatically) {
                    autoFilledPrompt = null
                }
            }

            override fun insertUpdate(event: DocumentEvent) = onChanged()

            override fun removeUpdate(event: DocumentEvent) = onChanged()

            override fun changedUpdate(event: DocumentEvent) = onChanged()
        })

        val promptScroll = createTextScroll(promptArea, preferredHeight = 176, centerContent = true)
        val resultsScroll = createResultsScroll(preferredHeight = 320)
        val graphSurface = JPanel(BorderLayout()).apply {
            isOpaque = true
            background = Palette.canvas
            alignmentX = LEFT_ALIGNMENT
            maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
            add(graphPanel, BorderLayout.CENTER)
        }
        val resultsViewLayout = CardLayout()
        val resultsViewCards = JPanel(resultsViewLayout).apply {
            isOpaque = true
            background = Palette.canvas
            alignmentX = LEFT_ALIGNMENT
            maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
            add(resultsScroll, "summary")
            add(graphSurface, "graph")
        }
        val animatedResultsViewport = SnapshotTransitionPanel(resultsViewCards, Palette.canvas).apply {
            isOpaque = true
            background = Palette.canvas
            alignmentX = LEFT_ALIGNMENT
            maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
        }
        val summaryTabButton = ResultsTabButton("Summary", selected = true)
        val graphTabButton = ResultsTabButton("Search Graph", selected = false)
        var currentResultsTab = "summary"
        fun selectResultsTab(tab: String) {
            if (currentResultsTab == tab) {
                return
            }
            summaryTabButton.isSelected = tab == "summary"
            graphTabButton.isSelected = tab == "graph"
            animatedResultsViewport.animateSwap {
                resultsViewLayout.show(resultsViewCards, tab)
                resultsViewCards.revalidate()
                resultsViewCards.repaint()
            }
            currentResultsTab = tab
        }
        summaryTabButton.addActionListener { selectResultsTab("summary") }
        graphTabButton.addActionListener { selectResultsTab("graph") }
        val resultsTabBar = JPanel(GridLayout(1, 2, 10, 0)).apply {
            isOpaque = false
            alignmentX = LEFT_ALIGNMENT
            maximumSize = Dimension(Int.MAX_VALUE, 58)
            add(summaryTabButton)
            add(graphTabButton)
        }
        val resultsDeck = RoundedPanel(
            fill = Palette.surface,
            borderColor = Palette.borderSoft,
            arc = 22,
            shadowed = false,
        ).apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            alignmentX = LEFT_ALIGNMENT
            maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
            border = JBUI.Borders.empty(16, 12, 12, 12)
            add(resultsTabBar)
            add(verticalGap(10))
            add(
                RoundedPanel(
                    fill = Palette.canvas,
                    borderColor = Palette.borderSoft,
                    arc = 20,
                    shadowed = false,
                ).apply {
                    layout = BorderLayout()
                    alignmentX = LEFT_ALIGNMENT
                    maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
                    border = JBUI.Borders.empty(8)
                    add(animatedResultsViewport, BorderLayout.CENTER)
                }
            )
        }
        resultsViewLayout.show(resultsViewCards, "summary")

        val heroCard = cardPanel(hero = true).apply {
            add(titleLabel("Hyperreasoning"))
            add(verticalGap(8))
            add(descriptionLabel("Intelligent & Efficient Task Searching"))
            add(verticalGap(18))
            add(contextPanel("Live context", contextArea))
        }

        val promptCard = cardPanel().apply {
            add(sectionEyebrow("INSTRUCTION"))
            add(verticalGap(10))
            add(sectionTitle("Task prompt"))
            add(verticalGap(6))
            add(descriptionLabel("Prompt used to guide search through solution space."))
            add(verticalGap(14))
            add(promptScroll)
        }

        val actionsMetaRow = JPanel(BorderLayout(12, 0)).apply {
            isOpaque = false
            alignmentX = LEFT_ALIGNMENT
            add(JPanel(GridLayout(0, 1, 0, 6)).apply {
                isOpaque = false
                add(runTests)
                add(runHiddenTests)
            }, BorderLayout.WEST)
            add(JPanel(GridLayout(1, 2, 10, 0)).apply {
                isOpaque = false
                add(refreshHistoryButton)
                add(healthButton)
            }, BorderLayout.EAST)
        }
        val actionGrid = JPanel(GridLayout(0, 3, 12, 12)).apply {
            isOpaque = false
            alignmentX = LEFT_ALIGNMENT
            maximumSize = Dimension(Int.MAX_VALUE, 168)
            add(runRainbowButton)
            add(runHeuristicButton)
            add(runOneShotButton)
            add(runRandomButton)
            add(compareButton)
        }
        val actionCard = cardPanel().apply {
            add(sectionEyebrow("ACTIONS"))
            add(verticalGap(10))
            add(sectionTitle("Run a strategy"))
            add(verticalGap(6))
            add(descriptionLabel("Run one policy or compare Rainbow, heuristic, and a 1-shot LLM baseline."))
            add(verticalGap(14))
            add(actionsMetaRow)
            add(verticalGap(12))
            add(actionGrid)
        }

        val progressCard = cardPanel().apply {
            add(sectionEyebrow("STATUS"))
            add(verticalGap(10))
            add(sectionTitle("Live progress"))
            add(verticalGap(6))
            add(descriptionLabel("Phase, step, and latency feedback."))
            add(verticalGap(12))
            add(runStateChip)
            add(verticalGap(12))
            progressBar.alignmentX = LEFT_ALIGNMENT
            progressBar.maximumSize = Dimension(Int.MAX_VALUE, 28)
            add(progressBar)
        }

        val outputCard = cardPanel().apply {
            add(JPanel().apply {
                isOpaque = false
                layout = BoxLayout(this, BoxLayout.Y_AXIS)
                alignmentX = LEFT_ALIGNMENT
                maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
                add(sectionEyebrow("RESULT"))
                add(verticalGap(10))
                add(sectionTitle("Results"))
                add(verticalGap(6))
                add(descriptionLabel("Readable summaries and comparisons."))
            })
            add(verticalGap(14))
            add(resultsDeck)
        }

        listOf(heroCard, promptCard, actionCard, progressCard, outputCard).forEachIndexed { index, card ->
            card.alignmentX = LEFT_ALIGNMENT
            card.maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
            content.add(card)
            if (index != 4) {
                content.add(verticalGap(14))
            }
        }

        rootViewCards.add(JBScrollPane(content).apply {
            horizontalScrollBarPolicy = JBScrollPane.HORIZONTAL_SCROLLBAR_NEVER
            verticalScrollBarPolicy = JBScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED
            border = JBUI.Borders.empty()
            viewport.isOpaque = false
            isOpaque = false
        }, "main")
        rootViewCards.add(JBScrollPane(historyViewContent).apply {
            horizontalScrollBarPolicy = JBScrollPane.HORIZONTAL_SCROLLBAR_NEVER
            verticalScrollBarPolicy = JBScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED
            border = JBUI.Borders.empty()
            viewport.isOpaque = false
            isOpaque = false
        }, "history")
        add(rootViewCards, BorderLayout.CENTER)

        healthButton.addActionListener {
            graphPanel.clear()
            renderResults(statusCard("Backend", "Checking backend health...", emptyList()))
            statusService.refreshHealth { status ->
                graphPanel.clear()
                renderResults(statusCard("Backend", status, emptyList()))
            }
        }

        runRandomButton.addActionListener { runTask(project, policy = "random") }
        runHeuristicButton.addActionListener { runTask(project, policy = "heuristic") }
        runRainbowButton.addActionListener { runTask(project, policy = "rainbow") }
        runOneShotButton.addActionListener { runTask(project, policy = "oneshot") }
        compareButton.addActionListener { compareStrategies(project) }
        refreshHistoryButton.addActionListener { showHistoryView(project) }

        project.messageBus.connect(project).subscribe(
            FileEditorManagerListener.FILE_EDITOR_MANAGER,
            object : FileEditorManagerListener {
                override fun selectionChanged(event: FileEditorManagerEvent) {
                    val newFile = event.newFile ?: return
                    syncPromptFromTask(project, newFile)
                    updateCurrentContext(project, newFile)
                }
            },
        )
        ApplicationManager.getApplication().invokeLater {
            FileEditorManager.getInstance(project).selectedFiles.firstOrNull()?.let {
                syncPromptFromTask(project, it)
                updateCurrentContext(project, it)
            }
        }
        graphPanel.clear()
        renderResults(statusCard("Ready", "Choose a strategy to run the current task.", listOf("Task metadata auto-fills prompt and context.")))
    }

    private fun runTask(project: Project, policy: String) {
        val virtualFile = FileEditorManager.getInstance(project).selectedFiles.firstOrNull()
        if (virtualFile == null) {
            graphPanel.clear()
            renderResults(statusCard("No active file", "Open a task file to run search.", emptyList()))
            return
        }

        syncPromptFromTask(project, virtualFile)

        settings.state.runTests = runTests.isSelected
        settings.state.runHiddenTests = runHiddenTests.isSelected

        val payload = buildTaskRunRequestPayload(project, virtualFile)
        val request = buildTaskRunRequest(payload, policy)
        graphPanel.clear()
        renderResults(
            requestCard("Submitting ${displayPolicyName(policy)} run", payload, request.targetFiles),
            statusCard("Run status", "Waiting for backend response...", listOf("Progress will stream below while the task runs.")),
        )
        setRequestRunning(true, "Submitting ${displayPolicyName(policy)} request...")
        statusService.runTaskWithProgress(
            request,
            onProgress = { status ->
                updateProgress(status.progress, prefix = status.kind)
            },
        ) { result ->
            setRequestRunning(false, "Ready")
            result.fold(
                onSuccess = { response ->
                    progressBar.value = 100
                    progressBar.string = "Completed"
                    progressLabel.text = "Completed ${displayPolicyName(response.strategy.policy)} in ${response.strategy.steps} steps"
                    graphPanel.showStrategies(listOf(response))
                    val cards = mutableListOf<JComponent>()
                    cards += requestCard("Run context", payload, request.targetFiles)
                    cards += strategyResultCard(response, highlight = true)
                    cards += detailsCard(response)
                    acceptChangesCard(project, payload, response, highlight = true)?.let { cards += it }
                    renderResults(*cards.toTypedArray())
                },
                onFailure = { exc ->
                    progressBar.value = 0
                    progressBar.string = "Failed"
                    progressLabel.text = "Run failed"
                    graphPanel.clear()
                    renderResults(
                        requestCard("Run context", payload, request.targetFiles),
                        statusCard("Run failed", exc.message ?: "Backend run failed.", listOf("Check backend health and the configured checkpoint.")),
                    )
                }
            )
        }
    }

    private fun compareStrategies(project: Project) {
        val virtualFile = FileEditorManager.getInstance(project).selectedFiles.firstOrNull()
        if (virtualFile == null) {
            graphPanel.clear()
            renderResults(statusCard("No active file", "Open a task file to compare strategies.", emptyList()))
            return
        }

        syncPromptFromTask(project, virtualFile)

        settings.state.runTests = runTests.isSelected
        settings.state.runHiddenTests = runHiddenTests.isSelected

        val payload = buildTaskRunRequestPayload(project, virtualFile)
        val request = buildCompareStrategiesRequest(payload)
        graphPanel.clear()
        renderResults(
            requestCard("Submitting comparison", payload, request.targetFiles),
            statusCard("Compare status", "Evaluating Rainbow, heuristic, and a 1-shot LLM baseline.", listOf("Results will rank strategies automatically.")),
        )
        setRequestRunning(true, "Submitting comparison...")
        statusService.compareTaskWithProgress(
            request,
            onProgress = { status ->
                updateProgress(status.progress, prefix = status.kind)
            },
        ) { result ->
            setRequestRunning(false, "Ready")
            result.fold(
                onSuccess = { response ->
                    progressBar.value = 100
                    progressBar.string = "Completed"
                    progressLabel.text = "Completed ${response.strategies.size} strategies"
                    graphPanel.showStrategies(response.strategies, forceRecordedSelection = true)
                    renderComparisonResults(project, response, payload, request.targetFiles)
                },
                onFailure = { exc ->
                    progressBar.value = 0
                    progressBar.string = "Failed"
                    progressLabel.text = "Comparison failed"
                    graphPanel.clear()
                    renderResults(
                        requestCard("Run context", payload, request.targetFiles),
                        statusCard("Comparison failed", exc.message ?: "Backend compare failed.", listOf("Check backend health and the configured checkpoint.")),
                    )
                }
            )
        }
    }

    private fun setRequestRunning(running: Boolean, label: String) {
        runHeuristicButton.isEnabled = !running
        runRainbowButton.isEnabled = !running
        runOneShotButton.isEnabled = !running
        runRandomButton.isEnabled = !running
        compareButton.isEnabled = !running
        healthButton.isEnabled = !running
        refreshHistoryButton.isEnabled = !running
        promptArea.isEnabled = !running
        runTests.isEnabled = !running
        runHiddenTests.isEnabled = !running
        progressBar.isVisible = true
        progressBar.isIndeterminate = running
        progressBar.value = 0
        progressBar.string = if (running) "Starting..." else "Idle"
        progressLabel.text = label
        runStateChip.bump()
        progressBar.bump()
        lastProgressVisualSignature = null
    }

    private fun updateProgress(progress: JobProgressSnapshot, prefix: String) {
        val totalPolicies = max(1, progress.totalPolicies)
        val policyOffset = if (progress.currentPolicyIndex > 0) progress.currentPolicyIndex - 1 else 0
        val perPolicyBase = 100 / totalPolicies.toDouble()
        val policyFraction = if (progress.maxSteps > 0) {
            progress.currentStep.toDouble() / progress.maxSteps.toDouble()
        } else {
            0.0
        }
        val percent = ((policyOffset * perPolicyBase) + (policyFraction * perPolicyBase)).toInt().coerceIn(0, 99)
        progressBar.isVisible = true
        progressBar.isIndeterminate = progress.maxSteps <= 0 || (progress.currentStep == 0 && progress.phase !in setOf("done", "failed"))
        if (!progressBar.isIndeterminate) {
            progressBar.value = percent
        }
        progressBar.string = buildProgressString(progress)
        progressLabel.text = buildProgressLabel(prefix, progress)
        val visualSignature = listOf(
            progress.phase,
            progress.policy ?: "-",
            progress.currentPolicyIndex.toString(),
            progress.labelTier ?: "-",
        ).joinToString("|")
        if (visualSignature != lastProgressVisualSignature) {
            progressBar.bump()
            runStateChip.bump()
            lastProgressVisualSignature = visualSignature
        }
    }

    private fun buildProgressLabel(prefix: String, progress: JobProgressSnapshot): String {
        val policyText = progress.policy?.let { "policy=${displayPolicyName(it)}" } ?: "policy=-"
        val stepText = if (progress.maxSteps > 0) "step ${progress.currentStep}/${progress.maxSteps}" else "step -"
        val strategyText = if (progress.totalPolicies > 1) {
            "strategy ${max(1, progress.currentPolicyIndex)}/${progress.totalPolicies}"
        } else {
            "single strategy"
        }
        return "$prefix: ${progress.phase} | $policyText | $strategyText | $stepText | elapsed=${"%.1f".format(progress.elapsedS)}s"
    }

    private fun buildProgressString(progress: JobProgressSnapshot): String {
        val fragments = mutableListOf<String>()
        fragments += progress.phase
        progress.lastLlmLabel?.let { fragments += "llm=$it" }
        progress.lastLlmDurationS?.let { fragments += "llm ${"%.1f".format(it)}s" }
        progress.lastCompileDurationS?.let { fragments += "compile ${"%.1f".format(it)}s" }
        progress.lastTestDurationS?.let { fragments += "test ${"%.1f".format(it)}s" }
        progress.action?.let { fragments += it }
        progress.labelTier?.let { fragments += it }
        return fragments.joinToString(" | ")
    }

    private fun renderResults(vararg cards: JComponent) {
        resultsContent.removeAll()
        cards.forEachIndexed { index, card ->
            val animatedCard = AnimatedCardWrapper(
                child = card,
                delayMs = index * 55,
                emphasize = card.getClientProperty("hyper.highlightPulse") == true,
            ).apply {
                alignmentX = LEFT_ALIGNMENT
                maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
            }
            resultsContent.add(animatedCard)
            if (index != cards.lastIndex) {
                resultsContent.add(verticalGap(12))
            }
        }
        resultsContent.revalidate()
        resultsContent.repaint()
    }

    private fun statusCard(title: String, summary: String, bullets: List<String>): JComponent {
        return resultCard(
            eyebrow = "STATUS",
            title = title,
            summary = summary,
            bullets = bullets,
        )
    }

    private fun requestCard(title: String, payload: TaskPayload, targetFiles: List<String>): JComponent {
        return resultCard(
            eyebrow = "REQUEST",
            title = title,
            summary = "${payload.family} • ${payload.files.size} files",
            titleMonospace = true,
            summaryMonospace = true,
            metricsMonospace = true,
            bulletsMonospace = true,
            metrics = listOf(
                "Targets" to targetFiles.joinToString().ifBlank { "-" },
                "Visible test" to (payload.visibleTestFile ?: "-"),
                "Hidden tests" to if (payload.runHiddenTests) "enabled" else "off",
                "Prompt" to if (payload.usesTaskJsonPrompt) "task.json" else "override",
                "Hidden file" to if (payload.hiddenTestFile != null) "backend-only" else "-",
            ),
            bullets = listOfNotNull(
                payload.taskRoot?.let { "Task root: $it" },
            ),
        )
    }

    private fun strategyResultCard(
        response: com.hyperreasoning.intellij.backend.TaskRunResponse,
        highlight: Boolean,
        rank: Int? = null,
    ): JComponent {
        val title = buildString {
            rank?.let { append("#").append(it).append(" ") }
            append(displayPolicyName(response.strategy.policy))
        }
        val status = when {
            response.strategy.testsTotal > 0 && response.strategy.testsPassed == response.strategy.testsTotal -> "All tests passing"
            response.strategy.testsPassed > 0 -> "Partial test pass"
            response.strategy.compileSuccesses > 0 -> "Compiles successfully"
            else -> "No successful compile yet"
        }
        return resultCard(
            eyebrow = if (highlight) "BEST STRATEGY" else "STRATEGY",
            title = title,
            summary = status,
            metrics = listOf(
                "Reward" to formatDecimal(response.strategy.totalReward),
                "Steps" to response.strategy.steps.toString(),
                "Compile" to response.strategy.compileSuccesses.toString(),
                "Tests" to strategyTestCount(response),
                "Visible tests" to "${response.strategy.visibleTestsPassed}/${response.strategy.visibleTestsTotal}",
                "Hidden tests" to "${response.strategy.hiddenTestsPassed}/${response.strategy.hiddenTestsTotal}",
                "Visible candidates" to response.strategy.visiblePasses.toString(),
                "Hidden candidates" to response.strategy.hiddenPasses.toString(),
                "Elapsed" to formatMetricSeconds(response.strategy.elapsedS),
                "LLM calls" to response.strategy.llmRequests.toString(),
                "Prompt tokens" to response.strategy.promptTokens.toString(),
                "Completion tokens" to response.strategy.completionTokens.toString(),
                "Cache" to when {
                    response.cacheHit -> "hit"
                    response.runId != null -> "stored"
                    else -> "-"
                },
                "Cloud" to response.cloudStatus,
            ),
            bullets = listOfNotNull(
                response.strategy.hiddenTestPassed?.let { "Hidden tests: ${if (it) "pass" else "fail"}" },
                response.bestPlan?.get("strategy")?.toString()?.let { "Best plan: $it" },
            ),
            highlight = highlight,
        )
    }

    private fun detailsCard(response: com.hyperreasoning.intellij.backend.TaskRunResponse): JComponent {
        return resultCard(
            eyebrow = "DETAILS",
            title = "Search graph",
            summary = "Structure explored during the run.",
            metrics = listOf(
                "Root candidates" to response.rootCandidates.size.toString(),
                "Transitions" to response.transitions.size.toString(),
                "Nodes" to response.nodes.size.toString(),
                "Edges" to response.edges.size.toString(),
            ),
            bullets = listOfNotNull(
                response.bestCompiledFiles.keys.takeIf { it.isNotEmpty() }?.joinToString()?.let { "Compiled files: $it" },
            ),
        )
    }

    private fun strategyTestCount(response: com.hyperreasoning.intellij.backend.TaskRunResponse): String {
        return if (response.strategy.testsTotal > 0) {
            "${response.strategy.testsPassed}/${response.strategy.testsTotal}"
        } else {
            "-"
        }
    }

    private fun renderComparisonResults(
        project: Project,
        response: com.hyperreasoning.intellij.backend.CompareStrategiesResponse,
        payload: TaskPayload,
        targetFiles: List<String>,
    ) {
        val ranked = response.strategies.sortedWith(
            compareByDescending<com.hyperreasoning.intellij.backend.TaskRunResponse> { it.strategy.testsPassed }
                .thenByDescending { it.strategy.fractionTestsPassed }
                .thenByDescending { it.strategy.hiddenTestsPassed }
                .thenByDescending { it.strategy.visibleTestsPassed }
                .thenByDescending { it.strategy.compileSuccesses }
                .thenBy { it.strategy.llmRequests }
                .thenBy { it.strategy.elapsedS ?: Double.MAX_VALUE }
                .thenBy { it.strategy.totalTokens }
                .thenByDescending { it.strategy.totalReward }
        )
        val cards = mutableListOf<JComponent>()
        cards += requestCard("Comparison context", payload, targetFiles)
        cards += resultCard(
            eyebrow = "COMPARISON",
            title = "Strategy ranking",
            summary = "Ordered by test outcome first, then lower LLM calls, time, and tokens.",
            bullets = ranked.mapIndexed { index, item ->
                "#${index + 1} ${item.strategy.policy}: tests ${strategyTestCount(item)}, candidate passes visible ${item.strategy.visiblePasses}, hidden ${item.strategy.hiddenPasses}, calls ${item.strategy.llmRequests}, tokens ${item.strategy.totalTokens}, time ${formatMetricSeconds(item.strategy.elapsedS)}"
            },
        )
        ranked.forEachIndexed { index, item ->
            cards += strategyResultCard(item, highlight = index == 0, rank = index + 1)
            acceptChangesCard(project, payload, item, highlight = index == 0)?.let { cards += it }
        }
        renderResults(*cards.toTypedArray())
    }

    private fun showHistoryView(project: Project) {
        rootViewLayout.show(rootViewCards, "history")
        refreshHistory(project)
    }

    private fun showMainView() {
        rootViewLayout.show(rootViewCards, "main")
    }

    private fun refreshHistory(project: Project) {
        val context = currentClientContext(project)
        renderHistoryStatus(project, "Loading cached runs...")
        statusService.listRuns(context) { result ->
            result.fold(
                onSuccess = { response ->
                    renderHistoryItems(project, response.items, response.cloudEnabled, response.errors)
                },
                onFailure = { exc ->
                    renderHistoryStatus(project, exc.message ?: "Could not load run history.")
                },
            )
        }
    }

    private fun syncHistory(project: Project) {
        val context = currentClientContext(project)
        renderHistoryStatus(project, "Syncing cached runs...")
        statusService.syncRuns(context) { result ->
            result.fold(
                onSuccess = { response ->
                    val summary = if (response.cloudEnabled) {
                        "uploaded=${response.uploaded}, downloaded=${response.downloaded}, failed=${response.failed}"
                    } else {
                        "Supabase is not configured in the backend .env file."
                    }
                    renderHistoryStatus(project, summary)
                    refreshHistory(project)
                },
                onFailure = { exc ->
                    renderHistoryStatus(project, exc.message ?: "Could not sync run history.")
                },
            )
        }
    }

    private fun renderHistoryStatus(project: Project, message: String) {
        renderHistoryCards(
            historyToolbarCard(project, cloudEnabled = false),
            resultCard(
                eyebrow = "CACHE",
                title = "Past runs",
                summary = message,
            )
        )
    }

    private fun renderHistoryItems(
        project: Project,
        items: List<RunHistoryItem>,
        cloudEnabled: Boolean,
        errors: List<String>,
    ) {
        val cards = mutableListOf<JComponent>()
        cards += historyToolbarCard(project, cloudEnabled)
        if (items.isEmpty()) {
            cards +=
                resultCard(
                    eyebrow = "CACHE",
                    title = "No cached runs",
                    summary = if (cloudEnabled) "No local or cloud runs found for this project." else "No local runs found. Supabase is disabled.",
                    bullets = errors,
                )
        } else {
            items.forEach { item ->
                cards += historyItemCard(project, item)
            }
            if (errors.isNotEmpty()) {
                cards +=
                    resultCard(
                        eyebrow = "SYNC",
                        title = "Cloud warnings",
                        summary = errors.first(),
                        bullets = errors.drop(1),
                    )
            }
        }
        renderHistoryCards(*cards.toTypedArray())
    }

    private fun historyToolbarCard(project: Project, cloudEnabled: Boolean): JComponent {
        return resultCard(
            eyebrow = "CACHE",
            title = "Past runs",
            summary = if (cloudEnabled) "Showing local .hyper cache plus Supabase-backed runs." else "Showing local .hyper cache. Supabase is not configured.",
            metrics = listOf("Cloud" to if (cloudEnabled) "enabled" else "disabled"),
        ).apply {
            add(verticalGap(12))
            add(JPanel(GridLayout(1, 3, 10, 0)).apply {
                isOpaque = false
                alignmentX = LEFT_ALIGNMENT
                maximumSize = Dimension(Int.MAX_VALUE, 46)
                add(ActionButton("Back", ButtonTone.GHOST).apply {
                    addActionListener { showMainView() }
                })
                add(ActionButton("Refresh", ButtonTone.SECONDARY).apply {
                    addActionListener { refreshHistory(project) }
                })
                add(ActionButton("Sync Cloud", ButtonTone.GHOST).apply {
                    addActionListener { syncHistory(project) }
                })
            })
        }
    }

    private fun renderHistoryCards(vararg cards: JComponent) {
        historyViewContent.removeAll()
        cards.forEachIndexed { index, card ->
            card.alignmentX = LEFT_ALIGNMENT
            card.maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
            historyViewContent.add(card)
            if (index != cards.lastIndex) {
                historyViewContent.add(verticalGap(12))
            }
        }
        historyViewContent.revalidate()
        historyViewContent.repaint()
    }

    private fun historyItemCard(project: Project, item: RunHistoryItem): JComponent {
        return RoundedPanel(
            fill = Palette.surface,
            borderColor = Palette.borderSoft,
            arc = 20,
            shadowed = false,
        ).apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            border = JBUI.Borders.empty(12, 14, 12, 14)
            add(sectionEyebrow("${displayPolicyName(item.policy)} • ${item.createdAt}"))
            add(verticalGap(6))
            add(scrollableText(item.promptPreview, Palette.textPrimary, Font.BOLD, 12f, 44))
            add(verticalGap(8))
            add(
                metricGrid(
                    listOf(
                        "Visible" to item.visiblePasses.toString(),
                        "Compile" to item.compileSuccesses.toString(),
                        "Local" to item.localStatus,
                        "Cloud" to item.cloudStatus,
                    )
                )
            )
            add(verticalGap(10))
            val loadLabel = if (item.policy == "comparison") "Load comparison" else "Load cached run"
            add(ActionButton(loadLabel, ButtonTone.PRIMARY).apply {
                alignmentX = LEFT_ALIGNMENT
                maximumSize = Dimension(Int.MAX_VALUE, 42)
                addActionListener { loadCachedRun(project, item) }
            })
        }
    }

    private fun loadCachedRun(project: Project, item: RunHistoryItem) {
        progressBar.isVisible = true
        progressBar.isIndeterminate = true
        progressBar.string = "Loading cache..."
        progressLabel.text = "Loading cached ${displayPolicyName(item.policy)} run"
        statusService.loadRun(item.runId, currentClientContext(project)) { result ->
            progressBar.isIndeterminate = false
            result.fold(
                onSuccess = { loaded ->
                    val comparison = loaded.compareResult
                    if (comparison != null) {
                        loadCachedComparison(project, loaded, comparison)
                        return@fold
                    }
                    val response = loaded.result ?: error("Cached task run did not include a task result.")
                    val payload = payloadFromCachedRun(loaded)
                    val request = loaded.request ?: error("Cached task run did not include its request.")
                    isUpdatingPromptProgrammatically = true
                    try {
                        promptArea.text = request.prompt
                        autoFilledPrompt = request.prompt
                    } finally {
                        isUpdatingPromptProgrammatically = false
                    }
                    progressBar.value = 100
                    progressBar.string = "Loaded"
                    progressLabel.text = "Loaded ${displayPolicyName(response.strategy.policy)} from cache"
                    graphPanel.showStrategies(listOf(response), forceRecordedSelection = true)
                    showMainView()
                    val cards = mutableListOf<JComponent>()
                    cards += cachedRequestCard(loaded)
                    cards += strategyResultCard(response, highlight = true)
                    cards += detailsCard(response)
                    acceptChangesCard(project, payload, response, highlight = true)?.let { cards += it }
                    renderResults(*cards.toTypedArray())
                },
                onFailure = { exc ->
                    progressBar.value = 0
                    progressBar.string = "Load failed"
                    progressLabel.text = "Cache load failed"
                    renderHistoryStatus(project, exc.message ?: "Could not load cached run.")
                },
            )
        }
    }

    private fun loadCachedComparison(
        project: Project,
        loaded: RunLoadResponse,
        response: com.hyperreasoning.intellij.backend.CompareStrategiesResponse,
    ) {
        val request = loaded.compareRequest ?: error("Cached comparison did not include its request.")
        val payload = payloadFromCachedComparison(loaded)
        isUpdatingPromptProgrammatically = true
        try {
            promptArea.text = request.prompt
            autoFilledPrompt = request.prompt
        } finally {
            isUpdatingPromptProgrammatically = false
        }
        progressBar.value = 100
        progressBar.string = "Loaded"
        progressLabel.text = "Loaded cached comparison"
        graphPanel.showStrategies(response.strategies, forceRecordedSelection = true)
        showMainView()
        renderComparisonResults(project, response, payload, request.targetFiles)
    }

    private fun cachedRequestCard(loaded: RunLoadResponse): JComponent {
        return resultCard(
            eyebrow = "CACHE",
            title = "Cached run context",
            summary = "${loaded.item.family} • ${loaded.request?.files?.size ?: 0} files",
            titleMonospace = true,
            summaryMonospace = true,
            metricsMonospace = true,
            bulletsMonospace = true,
            metrics = listOf(
                "Run" to loaded.item.runId,
                "Targets" to (loaded.request?.targetFiles?.joinToString()?.ifBlank { "-" } ?: "-"),
                "Visible test" to (loaded.request?.visibleTestFile ?: "-"),
                "Hidden tests" to if (loaded.request?.runHiddenTests == true) "enabled" else "off",
                "Cloud" to loaded.item.cloudStatus,
            ),
            bullets = listOfNotNull(
                loaded.request?.clientContext?.taskRoot?.let { "Task root: $it" },
                loaded.request?.clientContext?.activeFile?.let { "Active file: $it" },
            ),
        )
    }

    private fun payloadFromCachedRun(loaded: RunLoadResponse): TaskPayload {
        val request = loaded.request ?: error("Cached task run did not include its request.")
        val context = request.clientContext
        val taskRoot = context?.taskRoot
        val fileHints = if (taskRoot != null) {
            request.files.keys.associateWith { Path.of(taskRoot).resolve(it).normalize().toString() }
        } else if (request.files.size == 1 && context?.activeFile != null) {
            mapOf(request.files.keys.first() to context.activeFile)
        } else {
            emptyMap()
        }
        return TaskPayload(
            clientContext = context ?: ClientContext(),
            prompt = request.prompt,
            files = request.files,
            filePathHints = fileHints,
            targetFiles = request.targetFiles,
            visibleTestFile = request.visibleTestFile,
            hiddenTestFile = request.hiddenTestFile,
            family = request.family,
            language = request.language,
            taskRoot = taskRoot,
            usesTaskJson = taskRoot != null,
            usesTaskJsonPrompt = false,
            runHiddenTests = request.runHiddenTests,
        )
    }

    private fun payloadFromCachedComparison(loaded: RunLoadResponse): TaskPayload {
        val request = loaded.compareRequest ?: error("Cached comparison did not include its request.")
        val context = request.clientContext
        val taskRoot = context?.taskRoot
        val fileHints = if (taskRoot != null) {
            request.files.keys.associateWith { Path.of(taskRoot).resolve(it).normalize().toString() }
        } else if (request.files.size == 1 && context?.activeFile != null) {
            mapOf(request.files.keys.first() to context.activeFile)
        } else {
            emptyMap()
        }
        return TaskPayload(
            clientContext = context ?: ClientContext(),
            prompt = request.prompt,
            files = request.files,
            filePathHints = fileHints,
            targetFiles = request.targetFiles,
            visibleTestFile = request.visibleTestFile,
            hiddenTestFile = request.hiddenTestFile,
            family = request.family,
            language = request.language,
            taskRoot = taskRoot,
            usesTaskJson = taskRoot != null,
            usesTaskJsonPrompt = false,
            runHiddenTests = request.runHiddenTests,
        )
    }

    private fun acceptChangesCard(
        project: Project,
        payload: TaskPayload,
        response: com.hyperreasoning.intellij.backend.TaskRunResponse,
        highlight: Boolean,
    ): JComponent? {
        if (response.bestCompiledFiles.isEmpty()) {
            return null
        }
        val files = response.bestCompiledFiles.keys.sorted()
        val card = RoundedPanel(
            fill = if (highlight) Palette.cardStrong else Palette.surface,
            borderColor = if (highlight) Palette.buttonPrimary else Palette.borderSoft,
            arc = 22,
            shadowed = false,
            fillSecondary = if (highlight) Palette.cardStrongAlt else null,
        ).apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            border = JBUI.Borders.empty(14, 16, 14, 16)
            alignmentX = LEFT_ALIGNMENT
            maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
            putClientProperty("hyper.highlightPulse", highlight)
        }

        fun renderBody(
            title: String,
            summary: String,
            accepted: Boolean = false,
        ) {
            card.removeAll()
            card.add(sectionEyebrow("CHANGES"))
            card.add(verticalGap(8))
            card.add(JBLabel(title).apply {
                foreground = Palette.textPrimary
                font = labelFont(Font.BOLD, 16f)
                alignmentX = LEFT_ALIGNMENT
                maximumSize = Dimension(Int.MAX_VALUE, preferredSize.height)
            })
            card.add(verticalGap(4))
            card.add(contentSurface(scrollableText(summary, Palette.textSecondary, Font.PLAIN, 12f, 34)))
            card.add(verticalGap(10))
            card.add(
                contentSurface(
                    JPanel().apply {
                        isOpaque = false
                        layout = BoxLayout(this, BoxLayout.Y_AXIS)
                        alignmentX = LEFT_ALIGNMENT
                        maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
                        files.forEachIndexed { index, path ->
                            add(scrollableText("• $path", Palette.textSecondary, Font.PLAIN, 11f, 28, monospace = true))
                            if (index != files.lastIndex) {
                                add(verticalGap(4))
                            }
                        }
                    }
                )
            )
            if (!accepted) {
                card.add(verticalGap(12))
                card.add(JPanel(GridLayout(1, 2, 10, 0)).apply {
                    isOpaque = false
                    alignmentX = LEFT_ALIGNMENT
                    maximumSize = Dimension(Int.MAX_VALUE, 48)
                    add(ActionButton("Accept changes", ButtonTone.PRIMARY).apply {
                        addActionListener {
                            val confirmed = Messages.showYesNoDialog(
                                project,
                                "Apply ${files.size} file change(s) from ${displayPolicyName(response.strategy.policy)}?\n\nYou can undo with ${undoShortcutLabel()}.",
                                "Apply Hyperreasoning Changes",
                                "Apply",
                                "Cancel",
                                null,
                            )
                            if (confirmed != Messages.YES) {
                                return@addActionListener
                            }
                            val error = applyCompiledFiles(project, payload, response)
                            if (error == null) {
                                renderBody(
                                    title = "Changes applied",
                                    summary = "Applied to the editor via an undoable write command. Use ${undoShortcutLabel()} to undo.",
                                    accepted = true,
                                )
                            } else {
                                Messages.showErrorDialog(project, error, "Apply Hyperreasoning Changes")
                            }
                        }
                    })
                    add(ActionButton("Reject", ButtonTone.GHOST).apply {
                        addActionListener {
                            renderBody(
                                title = "Changes dismissed",
                                summary = "No files were modified. Run again later if you want to re-apply this candidate.",
                                accepted = true,
                            )
                        }
                    })
                })
            }
            card.revalidate()
            card.repaint()
        }

        renderBody(
            title = "Apply ${displayPolicyName(response.strategy.policy)} changes",
            summary = "${files.size} file change(s) are ready to apply. Nothing is modified until you confirm.",
        )
        return card
    }

    private fun applyCompiledFiles(
        project: Project,
        payload: TaskPayload,
        response: com.hyperreasoning.intellij.backend.TaskRunResponse,
    ): String? {
        val fileSystem = LocalFileSystem.getInstance()
        val resolvedFiles = response.bestCompiledFiles.map { (relativePath, newContent) ->
            val absolutePath = resolveCompiledFilePath(payload, relativePath)
                ?: return "Could not resolve a project file for `$relativePath`."
            val virtualFile = fileSystem.refreshAndFindFileByPath(absolutePath)
                ?: return "Could not find file `$relativePath` at `$absolutePath`."
            Triple(relativePath, virtualFile, newContent)
        }
        val fileManager = FileDocumentManager.getInstance()
        try {
            WriteCommandAction.runWriteCommandAction(
                project,
                "Apply Hyperreasoning Changes",
                null,
                Runnable {
                    resolvedFiles.forEach { (_, virtualFile, newContent) ->
                        val document = fileManager.getDocument(virtualFile)
                            ?: error("Could not open document for `${virtualFile.path}`.")
                        document.setText(newContent)
                    }
                },
            )
            resolvedFiles.firstOrNull()?.second?.let { firstFile ->
                FileEditorManager.getInstance(project).openFile(firstFile, true)
            }
        } catch (exc: Exception) {
            return exc.message ?: "Failed to apply generated changes."
        }
        return null
    }

    private fun resolveCompiledFilePath(payload: TaskPayload, relativePath: String): String? {
        payload.filePathHints[relativePath]?.let { return it }
        payload.filePathHints[Path.of(relativePath).fileName.toString()]?.let { return it }
        payload.taskRoot?.let { taskRoot ->
            return Path.of(taskRoot).resolve(relativePath).normalize().toString()
        }
        return if (payload.filePathHints.size == 1) payload.filePathHints.values.first() else null
    }

    private fun undoShortcutLabel(): String {
        return if (System.getProperty("os.name").lowercase().contains("mac")) "Cmd+Z" else "Ctrl+Z"
    }

    private fun resultCard(
        eyebrow: String,
        title: String,
        summary: String,
        metrics: List<Pair<String, String>> = emptyList(),
        bullets: List<String> = emptyList(),
        highlight: Boolean = false,
        titleMonospace: Boolean = false,
        summaryMonospace: Boolean = false,
        metricsMonospace: Boolean = false,
        bulletsMonospace: Boolean = false,
    ): JComponent {
        return RoundedPanel(
            fill = if (highlight) Palette.cardStrong else Palette.surface,
            borderColor = if (highlight) Palette.buttonPrimary else Palette.borderSoft,
            arc = 22,
            shadowed = false,
            fillSecondary = if (highlight) Palette.cardStrongAlt else null,
        ).apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            border = JBUI.Borders.empty(14, 16, 14, 16)
            alignmentX = LEFT_ALIGNMENT
            maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
            add(sectionEyebrow(eyebrow))
            add(verticalGap(8))
            add(JBLabel(title).apply {
                foreground = Palette.textPrimary
                font = if (titleMonospace) dankMonoFont(Font.BOLD, 16f) else labelFont(Font.BOLD, 16f)
                alignmentX = LEFT_ALIGNMENT
                maximumSize = Dimension(Int.MAX_VALUE, preferredSize.height)
            })
            add(verticalGap(4))
            add(contentSurface(scrollableText(summary, Palette.textSecondary, Font.PLAIN, 12f, 34, monospace = summaryMonospace)))
            if (metrics.isNotEmpty()) {
                add(verticalGap(12))
                add(metricGrid(metrics, valueMonospace = metricsMonospace))
            }
            if (bullets.isNotEmpty()) {
                add(verticalGap(10))
                add(
                    contentSurface(
                        JPanel().apply {
                            isOpaque = false
                            layout = BoxLayout(this, BoxLayout.Y_AXIS)
                            alignmentX = LEFT_ALIGNMENT
                            maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
                            bullets.forEachIndexed { index, bullet ->
                                add(scrollableText("• $bullet", Palette.textSecondary, Font.PLAIN, 11f, 28, monospace = bulletsMonospace))
                                if (index != bullets.lastIndex) {
                                    add(verticalGap(4))
                                }
                            }
                        }
                    )
                )
            }
        }.apply {
            putClientProperty("hyper.highlightPulse", highlight)
        }
    }

    private fun metricGrid(metrics: List<Pair<String, String>>, valueMonospace: Boolean = false): JComponent {
        return JPanel(GridLayout(0, 2, 10, 10)).apply {
            isOpaque = false
            alignmentX = LEFT_ALIGNMENT
            maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
            metrics.forEach { (label, value) -> add(metricChip(label, value, valueMonospace = valueMonospace)) }
        }
    }

    private fun metricChip(label: String, value: String, valueMonospace: Boolean = false): JComponent {
        return RoundedPanel(
            fill = Palette.canvas,
            borderColor = Palette.borderSoft,
            arc = 18,
            shadowed = false,
        ).apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            border = JBUI.Borders.empty(10, 12, 10, 12)
            add(JBLabel(label.uppercase()).apply {
                foreground = Palette.textTertiary
                font = labelFont(Font.BOLD, 10f)
                alignmentX = LEFT_ALIGNMENT
            })
            add(verticalGap(4))
            add(JBLabel(value).apply {
                foreground = Palette.textPrimary
                font = if (valueMonospace) dankMonoFont(Font.PLAIN, 12f) else labelFont(Font.BOLD, 12f)
                alignmentX = LEFT_ALIGNMENT
            })
        }
    }

    private fun configureTextArea(area: JBTextArea, monospaced: Boolean) {
        area.lineWrap = true
        area.wrapStyleWord = true
        area.foreground = Palette.textPrimary
        area.caretColor = Palette.textPrimary
        area.selectionColor = Palette.selection
        area.selectedTextColor = Palette.textPrimary
        area.background = Palette.surface
        area.border = JBUI.Borders.empty(14, 16, 14, 16)
        area.font = if (monospaced) dankMonoFont(Font.PLAIN, 12f) else dankMonoFont(Font.PLAIN, 13f)
        area.isOpaque = false
        area.alignmentX = LEFT_ALIGNMENT
    }

    private fun configureContextArea(area: JBTextArea) {
        configureTextArea(area, false)
        area.isEditable = false
        area.isFocusable = false
        area.isRequestFocusEnabled = false
        area.caretColor = Palette.surface
        area.rows = 4
        area.columns = 1
        area.lineWrap = false
        area.wrapStyleWord = false
        area.border = JBUI.Borders.empty(12, 14, 12, 14)
        area.font = dankMonoFont(Font.PLAIN, 12f)
    }

    private fun configureCheckBox(checkBox: JBCheckBox) {
        checkBox.isOpaque = false
        checkBox.foreground = Palette.textPrimary
        checkBox.font = labelFont(Font.PLAIN, 12f)
        checkBox.border = JBUI.Borders.empty(0)
        checkBox.isFocusPainted = false
    }

    private fun createResultsScroll(preferredHeight: Int): JComponent {
        return JBScrollPane(resultsContent).apply {
            preferredSize = Dimension(100, preferredHeight)
            maximumSize = Dimension(Int.MAX_VALUE, preferredHeight + 36)
            border = JBUI.Borders.empty()
            viewport.isOpaque = true
            viewport.background = Palette.canvas
            isOpaque = true
            background = Palette.canvas
            alignmentX = LEFT_ALIGNMENT
            horizontalScrollBarPolicy = JBScrollPane.HORIZONTAL_SCROLLBAR_NEVER
            verticalScrollBarPolicy = JBScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED
        }
    }

    private fun createTextScroll(component: JComponent, preferredHeight: Int, centerContent: Boolean = false): JComponent {
        val viewportView = if (centerContent) {
            CenteringPanel(component)
        } else {
            component
        }
        val scroll = JBScrollPane().apply {
            preferredSize = Dimension(100, preferredHeight)
            maximumSize = Dimension(Int.MAX_VALUE, preferredHeight + 36)
            border = JBUI.Borders.empty()
            viewport.isOpaque = false
            isOpaque = false
            alignmentX = LEFT_ALIGNMENT
            viewport.view = viewportView
            horizontalScrollBarPolicy = JBScrollPane.HORIZONTAL_SCROLLBAR_NEVER
            verticalScrollBarPolicy = JBScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED
        }
        return surfacePanel().apply {
            layout = BorderLayout()
            maximumSize = Dimension(Int.MAX_VALUE, preferredHeight + 12)
            add(scroll, BorderLayout.CENTER)
        }
    }

    private fun cardPanel(hero: Boolean = false): RoundedPanel {
        return RoundedPanel(
            fill = if (hero) Palette.cardStrong else Palette.card,
            fillSecondary = if (hero) Palette.cardStrongAlt else null,
            borderColor = Palette.border,
            arc = if (hero) 30 else 26,
            shadowed = true,
            interactive = true,
        ).apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            border = JBUI.Borders.empty(if (hero) 22 else 18, if (hero) 22 else 18, if (hero) 20 else 18, if (hero) 22 else 18)
        }
    }

    private fun sectionEyebrow(text: String): JBLabel {
        return JBLabel(text).apply {
            foreground = Palette.textTertiary
            font = labelFont(Font.BOLD, 11f)
            alignmentX = LEFT_ALIGNMENT
            horizontalAlignment = SwingConstants.LEFT
        }
    }

    private fun sectionTitle(text: String): JBLabel {
        return JBLabel(text).apply {
            foreground = Palette.textPrimary
            font = labelFont(Font.BOLD, 18f)
            alignmentX = LEFT_ALIGNMENT
            horizontalAlignment = SwingConstants.LEFT
        }
    }

    private fun titleLabel(text: String): JBLabel {
        return JBLabel(text).apply {
            foreground = Palette.textPrimary
            font = labelFont(Font.BOLD, 28f)
            alignmentX = LEFT_ALIGNMENT
            horizontalAlignment = SwingConstants.LEFT
        }
    }

    private fun descriptionLabel(text: String): JComponent {
        return scrollableText(text, Palette.textSecondary, Font.PLAIN, 12f, 34)
    }

    private fun scrollableText(
        text: String,
        color: Color,
        fontStyle: Int,
        fontSize: Float,
        preferredHeight: Int,
        monospace: Boolean = false,
    ): JComponent {
        val area = JBTextArea(text).apply {
            isEditable = false
            isFocusable = false
            isRequestFocusEnabled = false
            lineWrap = false
            wrapStyleWord = false
            foreground = color
            caretColor = Palette.surface
            background = Palette.surface
            border = JBUI.Borders.empty(0, 0, 0, 0)
            font = if (monospace) dankMonoFont(fontStyle, fontSize) else labelFont(fontStyle, fontSize)
            isOpaque = false
            alignmentX = LEFT_ALIGNMENT
            rows = max(1, text.count { it == '\n' } + 1)
            columns = 1
        }
        return JBScrollPane(area).apply {
            preferredSize = Dimension(100, preferredHeight)
            maximumSize = Dimension(Int.MAX_VALUE, preferredHeight + 12)
            border = JBUI.Borders.empty()
            viewport.isOpaque = false
            isOpaque = false
            alignmentX = LEFT_ALIGNMENT
            horizontalScrollBarPolicy = JBScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED
            verticalScrollBarPolicy = if (text.contains('\n')) JBScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED else JBScrollPane.VERTICAL_SCROLLBAR_NEVER
        }
    }

    private fun infoChip(title: String, valueLabel: JBLabel): JComponent {
        valueLabel.foreground = Palette.textPrimary
        valueLabel.font = labelFont(Font.PLAIN, 12f)
        return RoundedPanel(
            fill = Palette.surface,
            borderColor = Palette.borderSoft,
            arc = 22,
            shadowed = false,
        ).apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            border = JBUI.Borders.empty(12, 14, 12, 14)
            alignmentX = LEFT_ALIGNMENT
            maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
            add(JBLabel(title.uppercase()).apply {
                foreground = Palette.textTertiary
                font = labelFont(Font.BOLD, 10f)
                alignmentX = LEFT_ALIGNMENT
            })
            add(verticalGap(6))
            add(valueLabel.apply { alignmentX = LEFT_ALIGNMENT })
        }
    }

    private fun contextPanel(title: String, area: JBTextArea): JComponent {
        val scroll = JBScrollPane(area).apply {
            preferredSize = Dimension(100, 110)
            maximumSize = Dimension(Int.MAX_VALUE, 132)
            border = JBUI.Borders.empty()
            viewport.isOpaque = false
            isOpaque = false
            alignmentX = LEFT_ALIGNMENT
            horizontalScrollBarPolicy = JBScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED
            verticalScrollBarPolicy = JBScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED
        }
        return RoundedPanel(
            fill = Palette.surface,
            borderColor = Palette.borderSoft,
            arc = 22,
            shadowed = false,
        ).apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            border = JBUI.Borders.empty(12, 14, 12, 14)
            alignmentX = LEFT_ALIGNMENT
            add(JBLabel(title.uppercase()).apply {
                foreground = Palette.textTertiary
                font = labelFont(Font.BOLD, 10f)
                alignmentX = LEFT_ALIGNMENT
            })
            add(verticalGap(8))
            add(scroll)
        }
    }

    private fun surfacePanel(): RoundedPanel {
        return RoundedPanel(
            fill = Palette.surface,
            borderColor = Palette.borderSoft,
            arc = 20,
            shadowed = false,
        ).apply {
            border = JBUI.Borders.empty(0, 14, 0, 14)
            alignmentX = LEFT_ALIGNMENT
        }
    }

    private fun contentSurface(content: JComponent): JComponent {
        return surfacePanel().apply {
            layout = BorderLayout()
            maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
            border = JBUI.Borders.empty(10, 14, 10, 14)
            add(content, BorderLayout.CENTER)
        }
    }

    private fun labelFont(style: Int, size: Float): Font {
        return (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, style, size.toInt())).deriveFont(style, size)
    }

    private fun dankMonoFont(style: Int, size: Float): Font {
        val preferred = Font("Dank Mono", style, size.toInt())
        val resolvedFamily = preferred.family
        val fallback = UIManager.getFont("TextArea.font") ?: Font(Font.MONOSPACED, style, size.toInt())
        val base = if (resolvedFamily.equals("Dialog", ignoreCase = true) || resolvedFamily.equals("Monospaced", ignoreCase = true)) {
            fallback
        } else {
            preferred
        }
        return base.deriveFont(style, size)
    }

    private fun verticalGap(height: Int): JComponent {
        return JPanel().apply {
            isOpaque = false
            preferredSize = Dimension(0, height)
            maximumSize = Dimension(Int.MAX_VALUE, height)
            alignmentX = LEFT_ALIGNMENT
        }
    }

    private fun buildTaskRunRequest(payload: TaskPayload, policy: String): TaskRunRequest {
        return TaskRunRequest(
            clientContext = payload.clientContext,
            prompt = payload.prompt,
            files = payload.files,
            targetFiles = payload.targetFiles,
            visibleTestFile = payload.visibleTestFile,
            hiddenTestFile = payload.hiddenTestFile,
            language = payload.language,
            family = payload.family,
            policy = policy,
            proposalSource = if (policy == "oneshot") "llm" else settings.state.proposalSource,
            runTests = runTests.isSelected,
            runHiddenTests = payload.runHiddenTests,
            checkpointPath = settings.state.checkpointPath.takeIf { it.isNotBlank() },
        )
    }

    private fun buildCompareStrategiesRequest(payload: TaskPayload): CompareStrategiesRequest {
        return CompareStrategiesRequest(
            clientContext = payload.clientContext,
            prompt = payload.prompt,
            files = payload.files,
            targetFiles = payload.targetFiles,
            visibleTestFile = payload.visibleTestFile,
            hiddenTestFile = payload.hiddenTestFile,
            language = payload.language,
            family = payload.family,
            proposalSource = settings.state.proposalSource,
            runTests = runTests.isSelected,
            runHiddenTests = payload.runHiddenTests,
            checkpointPath = settings.state.checkpointPath.takeIf { it.isNotBlank() },
            policies = listOf("heuristic", "rainbow", "oneshot"),
        )
    }

    private fun displayPolicyName(policy: String): String {
        return when (policy.lowercase()) {
            "oneshot" -> "1-Shot LLM"
            "rainbow" -> "Rainbow"
            "heuristic" -> "Heuristic"
            "random" -> "Random"
            else -> policy.replace('_', ' ').replaceFirstChar { it.uppercase() }
        }
    }

    private fun currentClientContext(project: Project): ClientContext {
        val virtualFile = FileEditorManager.getInstance(project).selectedFiles.firstOrNull()
        val taskRoot = virtualFile?.let { findTaskRoot(project, Path.of(it.path)) }
        return buildClientContext(project, virtualFile, taskRoot)
    }

    private fun buildClientContext(project: Project, virtualFile: VirtualFile?, taskRoot: Path?): ClientContext {
        val projectRoot = project.basePath
        val projectName = project.name
        return ClientContext(
            projectId = stableProjectId(projectRoot, projectName),
            projectName = projectName,
            projectRoot = projectRoot,
            taskRoot = taskRoot?.toString(),
            activeFile = virtualFile?.path,
        )
    }

    private fun stableProjectId(projectRoot: String?, projectName: String): String {
        val basis = projectRoot ?: projectName
        val digest = MessageDigest.getInstance("SHA-256").digest(basis.toByteArray(Charsets.UTF_8))
        return digest.joinToString("") { "%02x".format(it.toInt() and 0xff) }.take(24)
    }

    private fun buildTaskRunRequestPayload(project: Project, virtualFile: VirtualFile): TaskPayload {
        val taskRoot = findTaskRoot(project, Path.of(virtualFile.path))
        return if (taskRoot != null) {
            val clientContext = buildClientContext(project, virtualFile, taskRoot)
            val taskJson = taskRoot.resolve("task.json")
            val parsed = json.parseToJsonElement(Files.readString(taskJson)) as JsonObject
            val promptFromTask = parsed.stringValue("prompt").orEmpty()
            val promptText = promptArea.text.trim()
            val effectivePrompt = if (promptText.isBlank() || promptText == defaultPromptText) {
                promptFromTask
            } else {
                promptText
            }
            val files = collectTaskFiles(
                taskRoot,
                hiddenTestFile = parsed.stringValue("hidden_test_file"),
                includeHiddenTest = runHiddenTests.isSelected,
            )
            TaskPayload(
                clientContext = clientContext,
                prompt = effectivePrompt,
                files = files,
                filePathHints = files.keys.associateWith { taskRoot.resolve(it).normalize().toString() },
                targetFiles = parsed.stringList("target_files"),
                visibleTestFile = parsed.stringValue("visible_test_file"),
                hiddenTestFile = parsed.stringValue("hidden_test_file"),
                family = parsed.stringValue("family") ?: "custom_single_file",
                language = parsed.stringValue("language") ?: "python",
                taskRoot = taskRoot.toString(),
                usesTaskJson = true,
                usesTaskJsonPrompt = effectivePrompt == promptFromTask,
                runHiddenTests = runHiddenTests.isSelected,
            )
        } else {
            val editor = FileEditorManager.getInstance(project).selectedTextEditor
            val document = editor?.document ?: error("No active editor/file found.")
            TaskPayload(
                clientContext = buildClientContext(project, virtualFile, null),
                prompt = promptArea.text.trim(),
                files = mapOf(virtualFile.name to document.text),
                filePathHints = mapOf(virtualFile.name to virtualFile.path),
                targetFiles = listOf(virtualFile.name),
                visibleTestFile = null,
                hiddenTestFile = null,
                family = "custom_single_file",
                language = "python",
                taskRoot = null,
                usesTaskJson = false,
                usesTaskJsonPrompt = false,
                runHiddenTests = runHiddenTests.isSelected,
            )
        }
    }

    private fun syncPromptFromTask(project: Project, virtualFile: VirtualFile) {
        val currentPrompt = promptArea.text.trim()
        val canReplacePrompt = currentPrompt.isBlank() ||
            currentPrompt == defaultPromptText ||
            currentPrompt == autoFilledPrompt
        if (!canReplacePrompt) {
            return
        }
        val taskPrompt = taskPromptFromFile(project, virtualFile) ?: return
        isUpdatingPromptProgrammatically = true
        try {
            promptArea.text = taskPrompt
            autoFilledPrompt = taskPrompt
        } finally {
            isUpdatingPromptProgrammatically = false
        }
    }

    private fun taskPromptFromFile(project: Project, virtualFile: VirtualFile): String? {
        val taskRoot = findTaskRoot(project, Path.of(virtualFile.path)) ?: return null
        val taskJson = taskRoot.resolve("task.json")
        if (!Files.exists(taskJson)) {
            return null
        }
        val parsed = json.parseToJsonElement(Files.readString(taskJson)) as? JsonObject ?: return null
        return parsed.stringValue("prompt")?.takeIf { it.isNotBlank() }
    }

    private fun updateCurrentContext(project: Project, virtualFile: VirtualFile?) {
        if (virtualFile == null) {
            contextArea.text = "Context\nno active file"
            return
        }
        val taskRoot = findTaskRoot(project, Path.of(virtualFile.path))
        if (taskRoot == null) {
            contextArea.text = buildString {
                append("file: ").append(virtualFile.path).append("\n")
                append("mode: ad hoc")
            }
            return
        }
        val taskJson = taskRoot.resolve("task.json")
        val parsed = if (Files.exists(taskJson)) {
            json.parseToJsonElement(Files.readString(taskJson)) as? JsonObject
        } else {
            null
        }
        val family = parsed?.stringValue("family") ?: "unknown"
        val targets = parsed?.stringList("target_files").orEmpty()
        val relative = taskRoot.relativize(Path.of(virtualFile.path)).toString().replace('\\', '/')
        contextArea.text = buildString {
            append("file: ").append(relative).append("\n")
            append("family: ").append(family).append("\n")
            append("task root: ").append(taskRoot).append("\n")
            append("targets: ")
            if (targets.isEmpty()) {
                append("-")
            } else {
                append(targets.joinToString())
            }
        }
    }

    private fun formatMetricSeconds(value: Double?): String {
        return if (value == null) "-" else "%.1fs".format(value)
    }

    private fun formatDecimal(value: Double): String {
        return "%.2f".format(value)
    }

    private fun collectTaskFiles(taskRoot: Path, hiddenTestFile: String?, includeHiddenTest: Boolean): Map<String, String> {
        val files = linkedMapOf<String, String>()
        Files.walk(taskRoot).use { paths ->
            paths
                .filter { Files.isRegularFile(it) }
                .filter { path ->
                    val relative = taskRoot.relativize(path).toString().replace('\\', '/')
                    isTaskPayloadFile(relative, hiddenTestFile, includeHiddenTest)
                }
                .sorted()
                .forEach { path ->
                    val relative = taskRoot.relativize(path).toString().replace('\\', '/')
                    try {
                        files[relative] = Files.readString(path)
                    } catch (_: MalformedInputException) {
                        // Skip binary/cache artifacts that are not part of the task definition.
                    }
                }
        }
        return files
    }

    private fun isTaskPayloadFile(relativePath: String, hiddenTestFile: String?, includeHiddenTest: Boolean): Boolean {
        val fileName = Path.of(relativePath).fileName.toString()
        if (relativePath == "task.json") {
            return false
        }
        if (relativePath.startsWith("reference/") ||
            relativePath.startsWith("__pycache__/") ||
            relativePath.contains("/__pycache__/") ||
            relativePath.startsWith(".idea/") ||
            relativePath.startsWith(".hyper/") ||
            relativePath.startsWith(".git/")
        ) {
            return false
        }
        if (fileName == "README.md" || fileName.endsWith(".md") || fileName.endsWith(".pyc")) {
            return false
        }
        val isHiddenTest = relativePath == hiddenTestFile ||
            relativePath == "test_hidden.py" ||
            relativePath.endsWith("/test_hidden.py")
        if (!includeHiddenTest && isHiddenTest) {
            return false
        }
        return true
    }

    private fun findTaskRoot(project: Project, start: Path): Path? {
        val basePath = project.basePath?.let { Path.of(it).toAbsolutePath().normalize() }
        var current: Path? = start.parent?.toAbsolutePath()?.normalize()
        while (current != null) {
            if (Files.exists(current.resolve("task.json"))) {
                return current
            }
            if (basePath != null && current == basePath) {
                break
            }
            current = current.parent
        }
        return null
    }

    private fun JsonObject.stringList(key: String): List<String> {
        val value = this[key] as? JsonArray ?: return emptyList()
        return value.mapNotNull { (it as? JsonPrimitive)?.content }
    }

    private fun JsonObject.stringValue(key: String): String? {
        return (this[key] as? JsonPrimitive)?.content
    }

    private data class TaskPayload(
        val clientContext: ClientContext,
        val prompt: String,
        val files: Map<String, String>,
        val filePathHints: Map<String, String>,
        val targetFiles: List<String>,
        val visibleTestFile: String?,
        val hiddenTestFile: String?,
        val family: String,
        val language: String,
        val taskRoot: String?,
        val usesTaskJson: Boolean,
        val usesTaskJsonPrompt: Boolean,
        val runHiddenTests: Boolean,
    )

    private class ViewportWidthPanel : JPanel(), Scrollable {
        override fun getPreferredScrollableViewportSize(): Dimension = preferredSize

        override fun getScrollableUnitIncrement(
            visibleRect: Rectangle,
            orientation: Int,
            direction: Int,
        ): Int = 24

        override fun getScrollableBlockIncrement(
            visibleRect: Rectangle,
            orientation: Int,
            direction: Int,
        ): Int = if (orientation == SwingConstants.VERTICAL) visibleRect.height else visibleRect.width

        override fun getScrollableTracksViewportWidth(): Boolean = true

        override fun getScrollableTracksViewportHeight(): Boolean = false
    }

    private class RoundedPanel(
        private val fill: Color,
        private val borderColor: Color,
        private val arc: Int,
        private val shadowed: Boolean,
        private val fillSecondary: Color? = null,
        private val interactive: Boolean = false,
    ) : JPanel() {
        private var phase = Math.random() * 6.0
        private var hoverProgress = 0.0
        private var targetHoverProgress = 0.0
        private var pointerXPx = 0.0
        private var pointerYPx = 0.0
        private var targetPointerXPx = 0.0
        private var targetPointerYPx = 0.0
        private var pointerInside = false
        private val animationTimer = Timer(16) {
            phase += 0.028
            if (interactive) {
                val pointer = currentPointerPosition()
                val hoveredNow = pointer != null
                if (hoveredNow != pointerInside) {
                    pointerInside = hoveredNow
                    if (hoveredNow) {
                        setHovered(this@RoundedPanel)
                    } else {
                        clearHovered(this@RoundedPanel)
                    }
                }
                targetHoverProgress = if (hoveredNow) 1.0 else 0.0
                if (pointer != null) {
                    targetPointerXPx = pointer.x.toDouble()
                    targetPointerYPx = pointer.y.toDouble()
                } else {
                    targetPointerXPx = width / 2.0
                    targetPointerYPx = height / 2.0
                }
            }
            hoverProgress += (targetHoverProgress - hoverProgress) * 0.18
            pointerXPx += (targetPointerXPx - pointerXPx) * 0.2
            pointerYPx += (targetPointerYPx - pointerYPx) * 0.2
            repaint()
        }

        init {
            isOpaque = false
            if (interactive) {
                register(this)
            }
            animationTimer.start()
        }

        override fun paintComponent(graphics: Graphics) {
            val g2 = graphics.create() as Graphics2D
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            val width = width - 1
            val height = height - 1
            val hoverCenterX = if (width <= 0) 0.0 else pointerXPx.coerceIn(0.0, width.toDouble())
            val hoverCenterY = if (height <= 0) 0.0 else pointerYPx.coerceIn(0.0, height.toDouble())
            val tiltX = if (width <= 0) 0.0 else ((hoverCenterX / width.toDouble()) - 0.5) * hoverProgress
            val tiltY = if (height <= 0) 0.0 else ((hoverCenterY / height.toDouble()) - 0.5) * hoverProgress
            val siblingMuted = interactive && hasActiveHoveredSibling(this)
            if (shadowed) {
                g2.color = Palette.shadowBreathing(phase, if (siblingMuted) hoverProgress * 0.35 else hoverProgress)
                val shadowDx = (tiltX * 14.0).toInt()
                val shadowDy = (5 + hoverProgress * 6.0 + kotlin.math.abs(tiltY) * 10.0).toInt()
                g2.fillRoundRect(3 + shadowDx, shadowDy, width - 5, height - 8, arc, arc)
            }
            g2.paint = if (fillSecondary != null) {
                GradientPaint(0f, 0f, fill, width.toFloat(), height.toFloat(), fillSecondary)
            } else {
                fill
            }
            g2.fillRoundRect(0, 0, width, height - if (shadowed) 2 else 0, arc, arc)
            g2.paint = GradientPaint(
                ((phaseWave(phase) * width) - width * 0.25).toFloat(),
                0f,
                Palette.cardSheen,
                ((phaseWave(phase) * width) + width * 0.55).toFloat(),
                height.toFloat(),
                Palette.cardSheenSoft,
                true,
            )
            g2.fillRoundRect(0, 0, width, height - if (shadowed) 2 else 0, arc, arc)
            if (interactive) {
                val sheenRadius = (width * (0.28 + hoverProgress * 0.16)).toInt().coerceAtLeast(80)
                g2.paint = GradientPaint(
                    (hoverCenterX - sheenRadius * 0.45).toFloat(),
                    (hoverCenterY - sheenRadius * 0.25).toFloat(),
                    Palette.cardHoverSheen(hoverProgress),
                    (hoverCenterX + sheenRadius).toFloat(),
                    (hoverCenterY + sheenRadius * 0.45).toFloat(),
                    Palette.cardHoverSheenSoft,
                    true,
                )
                g2.fillRoundRect(0, 0, width, height - if (shadowed) 2 else 0, arc, arc)
            }
            if (siblingMuted) {
                g2.color = Palette.cardSiblingMute
                g2.fillRoundRect(0, 0, width, height - if (shadowed) 2 else 0, arc, arc)
            }
            g2.color = borderColor
            g2.stroke = BasicStroke((1f + hoverProgress.toFloat() * 0.55f))
            g2.drawRoundRect(0, 0, width, height - if (shadowed) 2 else 0, arc, arc)
            if (interactive && hoverProgress > 0.02) {
                g2.color = Palette.cardHoverEdge(hoverProgress)
                g2.drawRoundRect(1, 1, width - 2, height - if (shadowed) 4 else 2, arc - 2, arc - 2)
            }
            g2.dispose()
            super.paintComponent(graphics)
        }

        private fun currentPointerPosition(): java.awt.Point? {
            return try {
                if (!isShowing) {
                    null
                } else {
                    getMousePosition(true)
                }
            } catch (_: IllegalComponentStateException) {
                null
            }
        }

        companion object {
            private var hoveredPanel: WeakReference<RoundedPanel>? = null
            private val interactivePanels = mutableListOf<WeakReference<RoundedPanel>>()

            private fun register(panel: RoundedPanel) {
                interactivePanels += WeakReference(panel)
            }

            private fun currentHovered(): RoundedPanel? {
                val panel = hoveredPanel?.get()
                if (panel == null) {
                    hoveredPanel = null
                }
                return panel
            }

            private fun setHovered(panel: RoundedPanel) {
                hoveredPanel = WeakReference(panel)
                repaintInteractivePanels()
            }

            private fun clearHovered(panel: RoundedPanel) {
                if (currentHovered() === panel) {
                    hoveredPanel = null
                    repaintInteractivePanels()
                }
            }

            private fun hasActiveHoveredSibling(panel: RoundedPanel): Boolean {
                val hovered = currentHovered() ?: return false
                return hovered !== panel
            }

            private fun repaintInteractivePanels() {
                val iterator = interactivePanels.iterator()
                while (iterator.hasNext()) {
                    val panel = iterator.next().get()
                    if (panel == null) {
                        iterator.remove()
                    } else {
                        panel.repaint()
                    }
                }
            }
        }
    }

    private class ActionButton(text: String, private val tone: ButtonTone) : JButton(text) {
        private var hoverProgress = 0.0
        private var pressProgress = 0.0
        private var shimmerPhase = Math.random() * 6.0
        private val animationTimer = Timer(16) {
            shimmerPhase += 0.048
            hoverProgress += (((if (model.isRollover) 1.0 else 0.0) - hoverProgress) * 0.2)
            pressProgress += (((if (model.isPressed) 1.0 else 0.0) - pressProgress) * 0.3)
            repaint()
        }

        init {
            isOpaque = false
            isContentAreaFilled = false
            isFocusPainted = false
            isBorderPainted = false
            isRolloverEnabled = true
            cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
            horizontalAlignment = SwingConstants.CENTER
            font = (UIManager.getFont("Button.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 12f)
            foreground = when (tone) {
                ButtonTone.PRIMARY -> Palette.buttonPrimaryText
                ButtonTone.SECONDARY, ButtonTone.GHOST -> Palette.textPrimary
            }
            border = JBUI.Borders.empty(12, 14, 12, 14)
            animationTimer.start()
        }

        override fun paintComponent(graphics: Graphics) {
            val g2 = graphics.create() as Graphics2D
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            val fill = when (tone) {
                ButtonTone.PRIMARY -> when {
                    !isEnabled -> Palette.buttonDisabled
                    model.isPressed -> Palette.buttonPrimaryPressed
                    model.isRollover -> Palette.buttonPrimaryHover
                    else -> Palette.buttonPrimary
                }
                ButtonTone.SECONDARY -> when {
                    !isEnabled -> Palette.buttonDisabledSoft
                    model.isPressed -> Palette.buttonSecondaryPressed
                    model.isRollover -> Palette.buttonSecondaryHover
                    else -> Palette.buttonSecondary
                }
                ButtonTone.GHOST -> when {
                    !isEnabled -> Palette.buttonDisabledSoft
                    model.isPressed -> Palette.buttonGhostPressed
                    model.isRollover -> Palette.buttonGhostHover
                    else -> Palette.buttonGhost
                }
            }
            val lift = hoverProgress * 2.4 - pressProgress * 1.6
            g2.translate(0.0, -lift)
            val borderColor = when (tone) {
                ButtonTone.PRIMARY -> fill
                ButtonTone.SECONDARY -> Palette.border
                ButtonTone.GHOST -> Palette.borderSoft
            }
            g2.color = Palette.buttonShadow
            g2.fillRoundRect(1, 5, width - 3, height - 6, 18, 18)
            g2.color = fill
            g2.fillRoundRect(0, 0, width - 1, height - 1, 18, 18)
            g2.paint = GradientPaint(
                0f,
                0f,
                Palette.buttonTopGloss(hoverProgress),
                0f,
                (height * (0.55 + hoverProgress * 0.08)).toFloat(),
                Palette.buttonTopGlossSoft,
                true,
            )
            g2.fillRoundRect(0, 0, width - 1, height - 1, 18, 18)
            if (hoverProgress > 0.01) {
                val sheenCenter = phaseWave(shimmerPhase) * width
                val sheenWidth = width * (0.16 + hoverProgress * 0.08)
                g2.paint = GradientPaint(
                    (sheenCenter - sheenWidth).toFloat(),
                    0f,
                    Palette.buttonHoverSheen(hoverProgress),
                    (sheenCenter + sheenWidth).toFloat(),
                    height.toFloat(),
                    Palette.buttonHoverSheenSoft,
                    true,
                )
                g2.fillRoundRect(0, 0, width - 1, height - 1, 18, 18)
            }
            g2.color = if (hoverProgress > 0.01) Palette.buttonHoverEdge(borderColor, hoverProgress) else borderColor
            g2.drawRoundRect(0, 0, width - 1, height - 1, 18, 18)
            if (hoverProgress > 0.02) {
                g2.color = Palette.buttonHoverOutline(hoverProgress)
                g2.drawRoundRect(1, 1, width - 3, height - 3, 16, 16)
            }
            g2.dispose()
            super.paintComponent(graphics)
        }
    }

    private class ResultsTabButton(text: String, selected: Boolean) : JToggleButton(text, selected) {
        private var hoverProgress = 0.0
        private var selectionProgress = if (selected) 1.0 else 0.0
        private var shimmerPhase = Math.random() * 6.0
        private val animationTimer = Timer(16) {
            shimmerPhase += 0.042
            hoverProgress += (((if (model.isRollover) 1.0 else 0.0) - hoverProgress) * 0.22)
            selectionProgress += (((if (isSelected) 1.0 else 0.0) - selectionProgress) * 0.24)
            repaint()
        }

        init {
            isOpaque = false
            isContentAreaFilled = false
            isFocusPainted = false
            isBorderPainted = false
            isRolloverEnabled = true
            cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
            horizontalAlignment = SwingConstants.CENTER
            foreground = Palette.textPrimary
            font = (UIManager.getFont("Button.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 12f)
            border = JBUI.Borders.empty(14, 14, 10, 14)
            animationTimer.start()
        }

        override fun paintComponent(graphics: Graphics) {
            val g2 = graphics.create() as Graphics2D
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            val fill = when {
                isSelected -> Palette.resultsTabSelected
                model.isPressed -> Palette.resultsTabPressed
                model.isRollover -> Palette.resultsTabHover
                else -> Palette.resultsTab
            }
            val borderColor = if (isSelected) Palette.resultsTabSelectedBorder else Palette.resultsTabBorder
            val topInset = 3
            val drawHeight = (height - topInset - 2).coerceAtLeast(1)
            g2.color = Palette.buttonShadow
            g2.fillRoundRect(1, topInset + 4, width - 3, drawHeight - 4, 18, 18)
            g2.color = fill
            g2.fillRoundRect(0, topInset, width - 1, drawHeight, 18, 18)
            g2.paint = GradientPaint(
                ((phaseWave(shimmerPhase) * width) - width * 0.18).toFloat(),
                topInset.toFloat(),
                if (isSelected) Palette.resultsTabSelectedSheen else Palette.buttonTopGloss(hoverProgress),
                ((phaseWave(shimmerPhase) * width) + width * 0.42).toFloat(),
                (topInset + drawHeight).toFloat(),
                Palette.buttonTopGlossSoft,
                true,
            )
            g2.fillRoundRect(0, topInset, width - 1, drawHeight, 18, 18)
            g2.color = borderColor
            g2.drawRoundRect(0, topInset, width - 1, drawHeight, 18, 18)
            g2.dispose()
            foreground = if (isSelected) Palette.resultsTabSelectedText else Palette.textPrimary
            super.paintComponent(graphics)
        }
    }

    private class AnimatedCardWrapper(
        private val child: JComponent,
        delayMs: Int,
        private val emphasize: Boolean,
    ) : JPanel(null) {
        private val appearDelayMs = delayMs.toLong()
        private val introDurationMs = 220.0
        private val birthAtMs = System.currentTimeMillis()
        private var introProgress = 0.0
        private var hoverProgress = 0.0
        private var targetHoverProgress = 0.0
        private var pulseProgress = if (emphasize) 1.0 else 0.0
        private val animationTimer: Timer = Timer(16, null)

        init {
            isOpaque = true
            background = Palette.canvas
            add(child)
            val hoverHandler = object : java.awt.event.MouseAdapter() {
                override fun mouseEntered(event: java.awt.event.MouseEvent) {
                    targetHoverProgress = 1.0
                    if (!animationTimer.isRunning) {
                        animationTimer.start()
                    }
                }

                override fun mouseExited(event: java.awt.event.MouseEvent) {
                    targetHoverProgress = 0.0
                    if (!animationTimer.isRunning) {
                        animationTimer.start()
                    }
                }
            }
            addMouseListener(hoverHandler)
            animationTimer.addActionListener {
                val elapsed = (System.currentTimeMillis() - birthAtMs - appearDelayMs).coerceAtLeast(0)
                introProgress = (elapsed / introDurationMs).coerceIn(0.0, 1.0)
                hoverProgress += (targetHoverProgress - hoverProgress) * 0.2
                pulseProgress *= 0.92
                if (introProgress >= 1.0 && kotlin.math.abs(targetHoverProgress - hoverProgress) < 0.01 && pulseProgress < 0.01) {
                    pulseProgress = 0.0
                    if (targetHoverProgress == 0.0) {
                        animationTimer.stop()
                    }
                }
                repaint()
            }
            animationTimer.start()
        }

        override fun paintComponent(graphics: Graphics) {
            val g2 = graphics.create() as Graphics2D
            g2.color = background
            g2.fillRect(0, 0, width, height)
            g2.dispose()
            super.paintComponent(graphics)
        }

        override fun doLayout() {
            child.setBounds(0, 0, width, height)
        }

        override fun getPreferredSize(): Dimension = child.preferredSize

        override fun getMaximumSize(): Dimension = child.maximumSize

        override fun paintChildren(graphics: Graphics) {
            super.paintChildren(graphics)
        }

        override fun paint(graphics: Graphics) {
            super.paint(graphics)
            if (introProgress >= 1.0 && hoverProgress < 0.01 && pulseProgress < 0.01) {
                return
            }
            val g2 = graphics.create() as Graphics2D
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            g2.clipRect(0, 0, width, height)
            val eased = easeOutCubic(introProgress)
            if (introProgress < 1.0) {
                g2.composite = AlphaComposite.getInstance(
                    AlphaComposite.SRC_OVER,
                    (1.0 - eased).toFloat().coerceIn(0f, 1f),
                )
                g2.color = background
                g2.fillRoundRect(0, 0, width, height, 24, 24)
            }
            g2.composite = AlphaComposite.SrcOver
            if (hoverProgress > 0.01) {
                g2.color = Palette.cardHoverEdge((hoverProgress * 0.5).coerceIn(0.0, 1.0))
                g2.drawRoundRect(1, 1, width - 3, height - 3, 24, 24)
            }
            if (pulseProgress > 0.01) {
                g2.color = Palette.cardHoverEdge((pulseProgress * 0.75).coerceIn(0.0, 1.0))
                g2.drawRoundRect(2, 2, width - 5, height - 5, 22, 22)
            }
            g2.dispose()
        }
    }

    private class PulseInfoChip(title: String, private val valueLabel: JBLabel) : JPanel() {
        private var pulse = 0.0
        private val animationTimer = Timer(16) {
            pulse *= 0.9
            repaint()
        }

        init {
            isOpaque = false
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            border = JBUI.Borders.empty(12, 14, 12, 14)
            alignmentX = LEFT_ALIGNMENT
            maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
            add(JBLabel(title.uppercase()).apply {
                foreground = Palette.textTertiary
                font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 10f)
                alignmentX = LEFT_ALIGNMENT
            })
            add(JPanel().apply {
                isOpaque = false
                preferredSize = Dimension(0, 6)
                maximumSize = Dimension(Int.MAX_VALUE, 6)
                alignmentX = LEFT_ALIGNMENT
            })
            add(valueLabel.apply { alignmentX = LEFT_ALIGNMENT })
            animationTimer.start()
        }

        fun bump() {
            pulse = max(pulse, 1.0)
            repaint()
        }

        override fun paintComponent(graphics: Graphics) {
            val g2 = graphics.create() as Graphics2D
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            g2.color = Palette.surface
            g2.fillRoundRect(0, 0, width - 1, height - 1, 22, 22)
            if (pulse > 0.01) {
                g2.color = Palette.infoPulse(pulse)
                g2.fillRoundRect(0, 0, width - 1, height - 1, 22, 22)
            }
            g2.color = Palette.borderSoft
            g2.drawRoundRect(0, 0, width - 1, height - 1, 22, 22)
            g2.dispose()
            super.paintComponent(graphics)
        }
    }

    private class RoundedProgressBar : JProgressBar() {
        private var displayedFraction = 0.0
        private var targetFraction = 0.0
        private var pulse = 0.0
        private val animationTimer = Timer(16) {
            displayedFraction += (targetFraction - displayedFraction) * 0.18
            pulse *= 0.9
            repaint()
        }

        init {
            isOpaque = false
            border = JBUI.Borders.empty(0)
            foreground = Palette.progressFill
            animationTimer.start()
        }

        override fun setValue(n: Int) {
            super.setValue(n)
            val range = (maximum - minimum).coerceAtLeast(1)
            targetFraction = ((n - minimum).toDouble() / range.toDouble()).coerceIn(0.0, 1.0)
        }

        fun bump() {
            pulse = max(pulse, 1.0)
        }

        override fun paintComponent(graphics: Graphics) {
            val g2 = graphics.create() as Graphics2D
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            val width = width - 1
            val height = height - 1
            val arc = height
            g2.color = Palette.progressTrack
            g2.fillRoundRect(0, 0, width, height, arc, arc)
            if (pulse > 0.01) {
                g2.color = Palette.infoPulse(pulse * 0.8)
                g2.fillRoundRect(0, 0, width, height, arc, arc)
            }

            if (isIndeterminate) {
                val segmentWidth = max(width / 3, 56)
                val travel = width + segmentWidth
                val x = ((System.currentTimeMillis() / 8L) % travel).toInt() - segmentWidth
                g2.clipRect(0, 0, width, height)
                g2.paint = GradientPaint(0f, 0f, Palette.progressFill, width.toFloat(), 0f, Palette.progressFillSoft)
                g2.fillRoundRect(x, 0, segmentWidth, height, arc, arc)
            } else {
                val fillWidth = (width * displayedFraction).toInt().coerceAtLeast(0)
                if (fillWidth > 0) {
                    g2.paint = GradientPaint(0f, 0f, Palette.progressFill, width.toFloat(), 0f, Palette.progressFillSoft)
                    g2.fillRoundRect(0, 0, fillWidth, height, arc, arc)
                    val shimmerX = (((System.currentTimeMillis() / 11L) % (fillWidth + 80)) - 40).toInt()
                    g2.paint = GradientPaint(
                        shimmerX.toFloat(),
                        0f,
                        Palette.progressSheen,
                        (shimmerX + 72).toFloat(),
                        height.toFloat(),
                        Palette.progressSheenSoft,
                        true,
                    )
                    g2.fillRoundRect(0, 0, fillWidth, height, arc, arc)
                }
            }

            g2.color = Palette.borderSoft
            g2.drawRoundRect(0, 0, width, height, arc, arc)
            if (isStringPainted && string != null) {
                g2.color = Palette.textPrimary
                g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 11f)
                val metrics = g2.fontMetrics
                val textWidth = metrics.stringWidth(string)
                val x = (width - textWidth) / 2
                val y = (height + metrics.ascent - metrics.descent) / 2
                g2.drawString(string, x, y)
            }
            g2.dispose()
        }
    }

    private class SnapshotTransitionPanel(
        private val content: JComponent,
        backgroundColor: Color,
    ) : JPanel(BorderLayout()) {
        private var overlayImage: BufferedImage? = null
        private var overlayAlpha = 0f
        private var transitionStartedAtMs = 0L
        private val transitionDurationMs = 160.0
        private val animationTimer: Timer = Timer(16, null)

        init {
            isOpaque = true
            background = backgroundColor
            add(content, BorderLayout.CENTER)
            animationTimer.addActionListener {
                val elapsed = (System.currentTimeMillis() - transitionStartedAtMs).coerceAtLeast(0L).toDouble()
                val overlayProgress = (elapsed / transitionDurationMs).coerceIn(0.0, 1.0).toFloat()
                overlayAlpha = (1f - overlayProgress * 0.95f).coerceAtLeast(0f)
                if (overlayAlpha <= 0.01f) {
                    overlayAlpha = 0f
                    overlayImage = null
                    animationTimer.stop()
                }
                repaint()
            }
        }

        fun animateSwap(swap: () -> Unit) {
            overlayImage = captureSnapshot()
            swap()
            if (overlayImage != null) {
                overlayAlpha = 1f
                transitionStartedAtMs = System.currentTimeMillis()
                animationTimer.restart()
            } else {
                repaint()
            }
        }

        override fun paintChildren(graphics: Graphics) {
            super.paintChildren(graphics)
            val image = overlayImage ?: return
            if (overlayAlpha <= 0f) {
                return
            }
            val g2 = graphics.create() as Graphics2D
            g2.composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, overlayAlpha.coerceIn(0f, 1f))
            g2.drawImage(image, 0, 0, null)
            g2.dispose()
        }

        private fun captureSnapshot(): BufferedImage? {
            if (width <= 0 || height <= 0) {
                return null
            }
            val image = BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
            val g2 = image.createGraphics()
            g2.color = background
            g2.fillRect(0, 0, width, height)
            content.paint(g2)
            g2.dispose()
            return image
        }
    }

    private class CenteringPanel(private val child: JComponent) : JPanel(GridBagLayout()) {
        init {
            isOpaque = false
            val constraints = GridBagConstraints().apply {
                gridx = 0
                gridy = 0
                weightx = 1.0
                weighty = 1.0
                fill = GridBagConstraints.BOTH
                anchor = GridBagConstraints.CENTER
            }
            add(child, constraints)
        }
    }

    private enum class ButtonTone {
        PRIMARY,
        SECONDARY,
        GHOST,
    }

    private object Palette {
        val canvas = JBColor(Color(0xF4F5F7), Color(0x111315))
        val card = JBColor(Color(0xFCFCFD), Color(0x181B1E))
        val cardStrong = JBColor(Color(0xFFFFFF), Color(0x1B1F23))
        val cardStrongAlt = JBColor(Color(0xF3F4F6), Color(0x16191C))
        val surface = JBColor(Color(0xF7F8FA), Color(0x14171A))
        val border = JBColor(Color(0xE1E5EA), Color(0x2E343A))
        val borderSoft = JBColor(Color(0xE7EBF0), Color(0x262B31))
        val shadow = JBColor(Color(0x14000000, true), Color(0x26000000, true))
        val cardSheen = JBColor(Color(0x18FFFFFF, true), Color(0x12FFFFFF, true))
        val cardSheenSoft = JBColor(Color(0x00FFFFFF, true), Color(0x00FFFFFF, true))
        val textPrimary = JBColor(Color(0x121417), Color(0xF6F7F9))
        val textSecondary = JBColor(Color(0x646B73), Color(0xA1A8B1))
        val textTertiary = JBColor(Color(0x8B929B), Color(0x77808A))
        val selection = JBColor(Color(0xDDE7FF), Color(0x2A3B57))
        val buttonPrimary = JBColor(Color(0x111315), Color(0xF4F5F6))
        val buttonPrimaryHover = JBColor(Color(0x1D2125), Color(0xFFFFFF))
        val buttonPrimaryPressed = JBColor(Color(0x2A2F34), Color(0xE2E5E8))
        val buttonPrimaryText = JBColor(Color(0xFFFFFF), Color(0x111315))
        val buttonSecondary = JBColor(Color(0xF5F7F9), Color(0x20252A))
        val buttonSecondaryHover = JBColor(Color(0xECEFF3), Color(0x272D33))
        val buttonSecondaryPressed = JBColor(Color(0xE2E7EC), Color(0x313940))
        val buttonGhost = JBColor(Color(0xFFFFFF), Color(0x181C20))
        val buttonGhostHover = JBColor(Color(0xF5F7F9), Color(0x20252A))
        val buttonGhostPressed = JBColor(Color(0xECEFF3), Color(0x272D33))
        val buttonDisabled = JBColor(Color(0xBFC6CE), Color(0x535B63))
        val buttonDisabledSoft = JBColor(Color(0xEEF1F4), Color(0x1D2125))
        val resultsTab = JBColor(Color(0xF2F5F9), Color(0x1C2126))
        val resultsTabHover = JBColor(Color(0xE9EEF5), Color(0x252B31))
        val resultsTabPressed = JBColor(Color(0xDFE6EF), Color(0x2C343C))
        val resultsTabSelected = JBColor(Color(0xE7EEFC), Color(0x26354C))
        val resultsTabBorder = JBColor(Color(0xD7DFE8), Color(0x313941))
        val resultsTabSelectedBorder = JBColor(Color(0x6C92E8), Color(0x5F8DF2))
        val resultsTabSelectedText = JBColor(Color(0x18263F), Color(0xF5F8FF))
        val resultsTabSelectedSheen = JBColor(Color(0x3EFFFFFF, true), Color(0x20FFFFFF, true))
        val buttonShadow = JBColor(Color(0x18000000, true), Color(0x26000000, true))
        val progressTrack = JBColor(Color(0xEDF1F4), Color(0x20252A))
        val progressFill = JBColor(Color(0x111315), Color(0xF2F4F6))
        val progressFillSoft = JBColor(Color(0x4B5563), Color(0xB7BDC5))
        val progressSheen = JBColor(Color(0x42FFFFFF, true), Color(0x28FFFFFF, true))
        val progressSheenSoft = JBColor(Color(0x00FFFFFF, true), Color(0x00FFFFFF, true))

        fun shadowBreathing(phase: Double, hoverProgress: Double = 0.0): Color {
            val alpha = (18 + (phaseWave(phase) * 14) + hoverProgress * 26).toInt().coerceIn(0, 255)
            return JBColor(Color(0, 0, 0, alpha), Color(0, 0, 0, (alpha + 10).coerceIn(0, 255)))
        }

        fun cardHoverSheen(intensity: Double): Color {
            val alpha = (14 + intensity * 42).toInt().coerceIn(0, 255)
            return JBColor(Color(255, 255, 255, alpha), Color(255, 255, 255, (alpha * 0.62).toInt().coerceIn(0, 255)))
        }

        val cardHoverSheenSoft = JBColor(Color(255, 255, 255, 0), Color(255, 255, 255, 0))
        val cardSiblingMute = JBColor(Color(244, 247, 250, 28), Color(10, 12, 14, 22))

        fun cardHoverEdge(intensity: Double): Color {
            val alpha = (22 + intensity * 48).toInt().coerceIn(0, 255)
            return JBColor(Color(255, 255, 255, alpha), Color(255, 255, 255, (alpha * 0.72).toInt().coerceIn(0, 255)))
        }

        fun buttonTopGloss(intensity: Double): Color {
            val alpha = (20 + intensity * 22).toInt().coerceIn(0, 255)
            return JBColor(Color(255, 255, 255, alpha), Color(255, 255, 255, (alpha * 0.42).toInt().coerceIn(0, 255)))
        }

        val buttonTopGlossSoft = JBColor(Color(255, 255, 255, 0), Color(255, 255, 255, 0))

        fun buttonHoverSheen(intensity: Double): Color {
            val alpha = (8 + intensity * 22).toInt().coerceIn(0, 255)
            return JBColor(Color(255, 255, 255, alpha), Color(255, 255, 255, (alpha * 0.5).toInt().coerceIn(0, 255)))
        }

        val buttonHoverSheenSoft = JBColor(Color(255, 255, 255, 0), Color(255, 255, 255, 0))

        fun buttonHoverEdge(base: Color, intensity: Double): Color {
            return mix(base, JBColor(Color(0xFFFFFF), Color(0xFFFFFF)), 0.08 + intensity * 0.16)
        }

        fun buttonHoverOutline(intensity: Double): Color {
            val alpha = (18 + intensity * 32).toInt().coerceIn(0, 255)
            return JBColor(Color(255, 255, 255, alpha), Color(255, 255, 255, (alpha * 0.58).toInt().coerceIn(0, 255)))
        }

        fun infoPulse(intensity: Double): Color {
            val alpha = (8 + intensity * 30).toInt().coerceIn(0, 255)
            return JBColor(Color(116, 162, 255, alpha), Color(88, 138, 242, (alpha * 0.72).toInt().coerceIn(0, 255)))
        }
    }
}

private fun phaseWave(value: Double): Double {
    return (kotlin.math.sin(value) + 1.0) / 2.0
}

private fun easeOutCubic(value: Double): Double {
    val clamped = value.coerceIn(0.0, 1.0)
    return 1.0 - (1.0 - clamped) * (1.0 - clamped) * (1.0 - clamped)
}

private fun mix(base: Color, overlay: Color, amount: Double): Color {
    val clamped = amount.coerceIn(0.0, 1.0)
    fun channel(a: Int, b: Int): Int = (a + (b - a) * clamped).toInt().coerceIn(0, 255)
    return Color(
        channel(base.red, overlay.red),
        channel(base.green, overlay.green),
        channel(base.blue, overlay.blue),
        channel(base.alpha, overlay.alpha),
    )
}
