package com.hyperreasoning.intellij.toolwindow

import com.hyperreasoning.intellij.backend.RunDiagnostic
import com.hyperreasoning.intellij.backend.TaskRunResponse
import com.intellij.openapi.project.Project
import com.intellij.ui.JBColor
import com.intellij.ui.components.JBLabel
import com.intellij.ui.components.JBPanel
import com.intellij.ui.components.JBScrollPane
import com.intellij.ui.components.JBTextArea
import com.intellij.util.ui.JBUI
import java.awt.AlphaComposite
import java.awt.BasicStroke
import java.awt.BorderLayout
import java.awt.CardLayout
import java.awt.Color
import java.awt.Component
import java.awt.Cursor
import java.awt.Dimension
import java.awt.Font
import java.awt.GradientPaint
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.GridLayout
import java.awt.Point
import java.awt.Polygon
import java.awt.RenderingHints
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import java.awt.event.MouseWheelEvent
import java.awt.geom.Path2D
import java.awt.image.BufferedImage
import javax.swing.BoxLayout
import javax.swing.JButton
import javax.swing.JComponent
import javax.swing.JDialog
import javax.swing.JPanel
import javax.swing.JSplitPane
import javax.swing.JToggleButton
import javax.swing.SwingConstants
import javax.swing.SwingUtilities
import javax.swing.Timer
import javax.swing.UIManager
import javax.swing.JViewport
import javax.swing.plaf.basic.BasicSplitPaneDivider
import javax.swing.plaf.basic.BasicSplitPaneUI
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

class SearchGraphPanel(
    private val project: Project,
    private val mode: SearchGraphPanelMode = SearchGraphPanelMode.FULL_GRAPH,
) : JBPanel<JBPanel<*>>(BorderLayout()) {
    private val stateStore = SearchGraphStateStore()
    private val headerLabel = titleLabel(mode.title)
    private val subtitleLabel = bodyLabel(mode.emptySubtitle)
    private val strategySelector = JPanel(GridLayout(0, 4, 8, 8)).apply {
        isOpaque = false
        alignmentX = Component.LEFT_ALIGNMENT
    }
    private val fitButton = GraphActionButton("Fit View", compact = true) { canvas.fitToTree() }
    private val expandButton = GraphActionButton("⛶", compact = true) { showExpandedWindow() }.apply {
        toolTipText = "Full Screen"
    }
    private val viewModeButton = GraphActionButton("Guided Path", compact = true) {
        guidedPathEnabled = !guidedPathEnabled
        updateGuidedPathVisibility()
    }
    private val searchStatusLabel = metricLabel("Idle")
    private val totalNodesLabel = metricLabel("0 nodes")
    private val activeNodesLabel = metricLabel("0 active")
    private val prunedNodesLabel = metricLabel("0 pruned")
    private val bestNodeLabel = metricLabel("Best -")
    private val detailsArea = JBTextArea().apply {
        isEditable = false
        isFocusable = false
        lineWrap = true
        wrapStyleWord = true
        border = JBUI.Borders.empty(14)
        font = dankMonoFont(Font.PLAIN, 12f)
        foreground = GraphPalette.textPrimary
        background = GraphPalette.surface
        isOpaque = false
        text = mode.emptyDetails
    }
    private val detailsContent = JPanel().apply {
        isOpaque = false
        layout = BoxLayout(this, BoxLayout.Y_AXIS)
        border = JBUI.Borders.empty()
        add(detailsArea)
    }
    private val canvas = SearchGraphCanvas(
        onNodeSelected = { nodeId ->
            stateStore.selectNode(nodeId)
            refreshSnapshot()
        },
    )
    private val graphScrollPane = JBScrollPane(canvas).apply {
        border = JBUI.Borders.empty()
        viewport.isOpaque = false
        isOpaque = false
        horizontalScrollBarPolicy = JBScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED
        verticalScrollBarPolicy = JBScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED
    }
    private val timelinePanel = GuidedPathTimelinePanel { nodeId ->
        stateStore.selectNode(nodeId)
        refreshSnapshot()
    }
    private val timelineScrollPane = JBScrollPane(timelinePanel).apply {
        border = JBUI.Borders.empty()
        viewport.isOpaque = false
        isOpaque = false
        horizontalScrollBarPolicy = JBScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED
        verticalScrollBarPolicy = JBScrollPane.VERTICAL_SCROLLBAR_NEVER
        preferredSize = Dimension(100, 188)
        minimumSize = Dimension(100, 164)
    }
    private val timelineCard = roundedCard().apply {
        layout = BorderLayout()
        border = JBUI.Borders.empty(10)
        add(JBLabel("Guided decisions").apply {
            foreground = GraphPalette.textPrimary
            font = labelFont(Font.BOLD, 13f)
            border = JBUI.Borders.empty(2, 6, 8, 6)
        }, BorderLayout.NORTH)
        add(timelineScrollPane, BorderLayout.CENTER)
    }
    private val viewCardLayout = CardLayout()
    private val viewCards = JPanel(viewCardLayout).apply {
        isOpaque = false
    }
    private val animatedViewSurface = SpringSwapPanel(viewCards, GraphPalette.canvas).apply {
        minimumSize = Dimension(0, 0)
        maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
    }

    private var currentStrategies: List<TaskRunResponse> = emptyList()
    private var currentSubscription: SearchGraphEventSubscription? = null
    private var hasLiveGraphState: Boolean = false
    private var currentStrategyKey: String? = null
    private var guidedPathEnabled: Boolean = false
    private var activeViewKey: String = "tree"
    private var diagnosticsByBankId: Map<String, List<RunDiagnostic>> = emptyMap()
    private var currentSnapshot: SearchGraphSnapshot = stateStore.snapshot(
        showPruned = true,
        bestPathOnly = false,
    )

    init {
        background = GraphPalette.canvas
        border = JBUI.Borders.empty(8)
        layout = BorderLayout(0, 12)

        add(buildHeader(), BorderLayout.NORTH)
        add(buildBody(), BorderLayout.CENTER)
        rebuildStrategyButtons()
        refreshSnapshot()
    }

    fun showStrategies(strategies: List<TaskRunResponse>, forceRecordedSelection: Boolean = false) {
        currentStrategies = strategies.sortedWith(
            compareByDescending<TaskRunResponse> { it.strategy.testsPassed }
                .thenByDescending { it.strategy.fractionTestsPassed }
                .thenByDescending { it.strategy.hiddenTestsPassed }
                .thenByDescending { it.strategy.visibleTestsPassed }
                .thenByDescending { it.strategy.compileSuccesses }
                .thenBy { it.strategy.llmRequests }
                .thenBy { it.strategy.elapsedS ?: Double.MAX_VALUE }
                .thenBy { it.strategy.totalTokens }
                .thenByDescending { it.strategy.totalReward }
        )
        rebuildStrategyButtons()
        if (currentStrategies.isEmpty()) {
            clear()
            return
        }
        val availableKeys = currentStrategies.map { strategyKey(it.strategy.policy) }.toSet()
        if (forceRecordedSelection || currentStrategyKey !in availableKeys) {
            currentStrategyKey = availableKeys.firstOrNull()
        }
        if (forceRecordedSelection || !hasLiveGraphState) {
            val selected = currentStrategies.firstOrNull {
                strategyKey(it.strategy.policy) == currentStrategyKey
            } ?: currentStrategies.first()
            showStrategy(selected)
        }
    }

    fun showEventSource(
        runId: String,
        source: SearchGraphEventSource,
        subtitle: String,
        autoPlay: Boolean = true,
    ) {
        bindSource(runId = runId, source = source, autoPlay = autoPlay, subtitle = subtitle)
    }

    fun clear() {
        currentSubscription?.dispose()
        currentSubscription = null
        hasLiveGraphState = false
        currentStrategyKey = null
        diagnosticsByBankId = emptyMap()
        stateStore.clear()
        subtitleLabel.text = mode.emptySubtitle
        renderDetails(mode.emptyDetails, emptyList())
        refreshSnapshot()
        updatePlaybackButtons()
    }

    private fun buildHeader(): JComponent {
        return JPanel().apply {
            isOpaque = false
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            add(leftStack {
                add(headerLabel)
                add(verticalGap(4))
                add(subtitleLabel)
            })
            add(verticalGap(10))
            add(strategySelector)
            add(verticalGap(10))
            add(controlRow())
            add(verticalGap(10))
            add(legendRow())
            add(verticalGap(10))
            add(statusRow())
        }
    }

    private fun buildBody(): JComponent {
        val graphCard = roundedCard().apply {
            layout = BorderLayout()
            minimumSize = Dimension(0, 0)
            add(graphScrollPane, BorderLayout.CENTER)
        }
        viewCards.minimumSize = Dimension(0, 0)
        viewCards.removeAll()
        viewCards.add(graphCard, "tree")
        viewCards.add(timelineCard, "timeline")
        activeViewKey = if (guidedPathEnabled) "timeline" else "tree"
        viewCardLayout.show(viewCards, activeViewKey)
        val detailsCard = roundedCard().apply {
            preferredSize = Dimension(330, 100)
            minimumSize = Dimension(180, 100)
            layout = BorderLayout()
            add(JBLabel("Node details").apply {
                foreground = GraphPalette.textPrimary
                font = labelFont(Font.BOLD, 13f)
                border = JBUI.Borders.empty(12, 14, 0, 14)
            }, BorderLayout.NORTH)
            add(JBScrollPane(detailsContent).apply {
                border = JBUI.Borders.empty()
                viewport.isOpaque = false
                isOpaque = false
            }, BorderLayout.CENTER)
        }
        val splitPane = JSplitPane(JSplitPane.HORIZONTAL_SPLIT, animatedViewSurface, detailsCard).apply {
            isOpaque = false
            border = JBUI.Borders.empty()
            dividerSize = 12
            resizeWeight = 0.72
            setContinuousLayout(true)
            setDividerLocation(0.72)
            leftComponent.minimumSize = Dimension(0, 0)
            rightComponent.minimumSize = Dimension(180, 0)
            setOneTouchExpandable(false)
            ui = GraphSplitPaneUI()
        }
        return JPanel(BorderLayout()).apply {
            isOpaque = false
            add(splitPane, BorderLayout.CENTER)
        }
    }

    private fun controlRow(): JComponent {
        return JPanel(GridLayout(1, 3, 8, 0)).apply {
            isOpaque = false
            alignmentX = Component.LEFT_ALIGNMENT
            add(fitButton)
            add(expandButton)
            add(viewModeButton)
        }
    }

    private fun legendRow(): JComponent {
        return JPanel(GridLayout(0, 3, 8, 8)).apply {
            isOpaque = false
            alignmentX = Component.LEFT_ALIGNMENT
            add(LegendChip("Root", GraphPalette.root))
            add(LegendChip("Frontier", GraphPalette.bestPath))
            add(LegendChip("Solved", GraphPalette.success))
            add(LegendChip("Pruned", GraphPalette.pruned))
            add(LegendChip("Compile fail", GraphPalette.failedCompile))
            add(LegendChip("Test fail", GraphPalette.failedTest))
        }
    }

    private fun statusRow(): JComponent {
        return JPanel(GridLayout(1, 5, 8, 0)).apply {
            isOpaque = false
            alignmentX = Component.LEFT_ALIGNMENT
            add(StatusChip("Status", searchStatusLabel))
            add(StatusChip("Nodes", totalNodesLabel))
            add(StatusChip("Active", activeNodesLabel))
            add(StatusChip("Pruned", prunedNodesLabel))
            add(StatusChip("Best", bestNodeLabel))
        }
    }

    private fun leftStack(content: JPanel.() -> Unit): JComponent {
        return JPanel().apply {
            isOpaque = false
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            alignmentX = Component.LEFT_ALIGNMENT
            content()
        }
    }

    private fun rebuildStrategyButtons() {
        strategySelector.removeAll()
        currentStrategies.forEachIndexed { index, response ->
            val key = strategyKey(response.strategy.policy)
            val label = "${displayPolicyName(response.strategy.policy)}${if (index == 0) " • best" else ""}"
            strategySelector.add(GraphStrategyButton(label, selected = key == currentStrategyKey) {
                currentStrategyKey = key
                showStrategy(response)
            })
        }
        strategySelector.revalidate()
        strategySelector.repaint()
    }

    private fun showStrategy(response: TaskRunResponse) {
        val key = strategyKey(response.strategy.policy)
        currentStrategyKey = key
        diagnosticsByBankId = response.diagnostics
            .filter { it.bankId != null }
            .groupBy { it.bankId.orEmpty() }
        val finalizedEvents = SearchGraphBackendAdapter.decode(response.searchGraphEvents)
        if (finalizedEvents.isNotEmpty()) {
            showCompletedStrategyGraph(finalizedEvents)
            return
        }
        showRecordedStrategy(response)
    }

    private fun showCompletedStrategyGraph(events: List<SearchGraphEvent>) {
        currentSubscription?.dispose()
        currentSubscription = null
        hasLiveGraphState = true
        stateStore.clear()
        renderDetails(mode.emptyDetails, emptyList())
        rebuildStrategyButtons()
        events.forEach(stateStore::apply)
        refreshSnapshot()
        updatePlaybackButtons()
        canvas.fitToTree()
    }

    private fun showRecordedStrategy(response: TaskRunResponse) {
        currentSubscription?.dispose()
        currentSubscription = null
        hasLiveGraphState = false
        currentStrategyKey = strategyKey(response.strategy.policy)
        stateStore.clear()
        subtitleLabel.text = "Recorded search tree derived from the completed run."
        renderDetails(mode.emptyDetails, emptyList())
        rebuildStrategyButtons()
        val runId = "recorded-${response.strategy.policy}"
        val events = SearchGraphEventSources.recordedRunEvents(runId, displayPolicyName(response.strategy.policy), response)
        events.forEach(stateStore::apply)
        refreshSnapshot()
        updatePlaybackButtons()
        canvas.fitToTree()
    }

    fun showSnapshot(
        snapshot: SearchGraphSnapshot,
        strategies: List<TaskRunResponse> = emptyList(),
        strategyKey: String? = null,
        guidedPathEnabled: Boolean = this.guidedPathEnabled,
        diagnosticsByBankId: Map<String, List<RunDiagnostic>> = this.diagnosticsByBankId,
    ) {
        currentSubscription?.dispose()
        currentSubscription = null
        currentStrategies = strategies
        currentStrategyKey = strategyKey
        this.diagnosticsByBankId = diagnosticsByBankId
        hasLiveGraphState = true
        this.guidedPathEnabled = guidedPathEnabled
        stateStore.loadSnapshot(snapshot)
        rebuildStrategyButtons()
        currentSnapshot = snapshot
        subtitleLabel.text = if (snapshot.runId == null) mode.emptySubtitle else mode.subtitleFor(snapshot)
        canvas.setSnapshot(snapshot)
        timelinePanel.setSnapshot(snapshot)
        updateStatusStrip(snapshot)
        updateDetails(snapshot)
        updateGuidedPathVisibility()
        updatePlaybackButtons()
        canvas.fitToTree()
    }

    private fun bindSource(
        runId: String,
        source: SearchGraphEventSource,
        autoPlay: Boolean,
        subtitle: String,
    ) {
        currentSubscription?.dispose()
        stateStore.clear()
        hasLiveGraphState = false
        subtitleLabel.text = subtitle
        diagnosticsByBankId = emptyMap()
        renderDetails(mode.emptyDetails, emptyList())
        currentSubscription = source.subscribeToSearchGraphEvents(runId) { event ->
            stateStore.apply(event)
            refreshSnapshot()
        }
        currentSubscription?.reset()
        updatePlaybackButtons()
        if (autoPlay) {
            currentSubscription?.play()
        }
    }

    private fun updatePlaybackButtons() {
        val enabled = currentSubscription?.supportsPlayback == true
        listOf(fitButton, expandButton).forEach { it.isEnabled = true }
    }

    private fun refreshSnapshot() {
        currentSnapshot = snapshot()
        subtitleLabel.text = if (currentSnapshot.runId == null) mode.emptySubtitle else mode.subtitleFor(currentSnapshot)
        canvas.setSnapshot(currentSnapshot)
        timelinePanel.setSnapshot(currentSnapshot)
        updateStatusStrip(currentSnapshot)
        updateDetails(currentSnapshot)
        updateGuidedPathVisibility()
    }

    private fun updateStatusStrip(snapshot: SearchGraphSnapshot) {
        searchStatusLabel.text = snapshot.lifecycleStatus.name.lowercase().replace('_', ' ')
        totalNodesLabel.text = if (snapshot.nodes.size == snapshot.totalNodes) {
            snapshot.totalNodes.toString()
        } else {
            "${snapshot.nodes.size}/${snapshot.totalNodes}"
        }
        activeNodesLabel.text = snapshot.activeNodes.toString()
        prunedNodesLabel.text = snapshot.prunedNodes.toString()
        bestNodeLabel.text = snapshot.bestNodeId ?: "-"
    }

    private fun updateDetails(snapshot: SearchGraphSnapshot) {
        val selected = snapshot.nodes.firstOrNull { it.id == snapshot.selectedNodeId }
            ?: snapshot.nodes.firstOrNull { it.id == snapshot.bestNodeId }
            ?: snapshot.nodes.firstOrNull()
        if (selected == null) {
            renderDetails(mode.emptyDetails, emptyList())
            return
        }
        val detailsText = buildString {
            append("title: ").append(selected.title).append('\n')
            append("id: ").append(selected.id).append('\n')
            selected.parentId?.let { append("parent: ").append(it).append('\n') }
            append("depth: ").append(selected.depth).append('\n')
            append("status: ").append(selected.status.name.lowercase().replace('_', ' ')).append('\n')
            selected.childIndex?.let { index ->
                append("child slot: ").append(index)
                selected.childCount?.let { count -> append(" of ").append(count) }
                append('\n')
            }
            selected.score?.let { append("score: ").append(formatScore(it)).append('\n') }
            selected.heuristicScore?.let { append("heuristic: ").append(formatScore(it)).append('\n') }
            selected.qValue?.let { append("q-value: ").append(formatScore(it)).append('\n') }
            selected.rank?.let { append("rank: ").append(it).append('\n') }
            append("generation: ").append(selected.createdOrder).append('\n')
            selected.createdAtLabel?.let { append("timestamp: ").append(it).append('\n') }
            selected.compileStatus?.let { append("compile: ").append(it).append('\n') }
            selected.testStatus?.let { append("tests: ").append(it).append('\n') }
            selected.runtimeStatus?.let { append("runtime: ").append(it).append('\n') }
            selected.terminalSummary?.let { append("terminal: ").append(it).append('\n') }
            if (selected.dslSummary.isNotBlank()) {
                append("\nDSL\n").append(selected.dslSummary).append('\n')
            }
            if (selected.patchSummary.isNotBlank()) {
                append("\npatch\n").append(selected.patchSummary).append('\n')
            }
            if (selected.shortSummary.isNotBlank()) {
                append("\nsummary\n").append(selected.shortSummary).append('\n')
            }
            if (selected.rationaleSummary.isNotBlank()) {
                append("\nnotes\n").append(selected.rationaleSummary).append('\n')
            }
        }
        renderDetails(detailsText, diagnosticsByBankId[selected.id].orEmpty())
    }

    private fun renderDetails(text: String, diagnostics: List<RunDiagnostic>) {
        detailsArea.text = text
        detailsContent.removeAll()
        detailsContent.add(detailsArea.apply {
            alignmentX = Component.LEFT_ALIGNMENT
            maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
        })
        diagnosticsDropdown(diagnostics)?.let { dropdown ->
            detailsContent.add(verticalGap(8))
            detailsContent.add(dropdown)
        }
        detailsContent.revalidate()
        detailsContent.repaint()
    }

    private fun diagnosticsDropdown(diagnostics: List<RunDiagnostic>): JComponent? {
        if (diagnostics.isEmpty()) {
            return null
        }
        val body = JPanel().apply {
            isOpaque = false
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            alignmentX = Component.LEFT_ALIGNMENT
            isVisible = false
        }
        val toggle = JToggleButton("Diagnostics > ${diagnosticsSummary(diagnostics)}").apply {
            isOpaque = false
            isContentAreaFilled = false
            isBorderPainted = false
            isFocusPainted = false
            horizontalAlignment = SwingConstants.LEFT
            foreground = GraphPalette.textPrimary
            font = labelFont(Font.BOLD, 12f)
            cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
            border = JBUI.Borders.empty(8, 2, 8, 2)
            addActionListener {
                body.isVisible = isSelected
                text = "${if (isSelected) "Diagnostics v" else "Diagnostics >"} ${diagnosticsSummary(diagnostics)}"
                body.revalidate()
                body.repaint()
            }
        }
        diagnostics.forEachIndexed { index, diagnostic ->
            body.add(diagnosticRow(index + 1, diagnostic))
            if (index != diagnostics.lastIndex) {
                body.add(verticalGap(10))
            }
        }
        return JPanel().apply {
            isOpaque = false
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            border = JBUI.Borders.empty(8, 4, 4, 4)
            alignmentX = Component.LEFT_ALIGNMENT
            maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
            add(toggle)
            add(body)
        }
    }

    private fun diagnosticRow(index: Int, diagnostic: RunDiagnostic): JComponent {
        return JPanel().apply {
            isOpaque = false
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            alignmentX = Component.LEFT_ALIGNMENT
            add(bodyLabel(buildDiagnosticTitle(index, diagnostic)).apply {
                font = dankMonoFont(Font.BOLD, 12f)
                alignmentX = Component.LEFT_ALIGNMENT
            })
            add(verticalGap(6))
            add(JPanel(GridLayout(0, 2, 8, 8)).apply {
                isOpaque = false
                alignmentX = Component.LEFT_ALIGNMENT
                maximumSize = Dimension(Int.MAX_VALUE, Int.MAX_VALUE)
                listOf(
                    "status" to diagnostic.status,
                    "compile" to diagnostic.compileSuccess.renderNullableBool(),
                    "visible" to diagnostic.visibleTestPassed.renderNullableBool(),
                    "hidden" to diagnostic.hiddenTestPassed.renderNullableBool(),
                    "bank" to (diagnostic.bankId ?: "-"),
                    "targets" to diagnostic.targetFiles.joinToString().ifBlank { "-" },
                ).forEach { (label, value) ->
                    add(diagnosticMetric(label, value))
                }
            })
            diagnosticTerminalSections(diagnostic).forEach { (label, output) ->
                add(verticalGap(8))
                add(terminalBlock(label, output))
            }
        }
    }

    private fun diagnosticMetric(label: String, value: String): JComponent {
        return JPanel().apply {
            isOpaque = false
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            add(bodyLabel(label.uppercase()).apply {
                foreground = GraphPalette.textMuted
                font = labelFont(Font.BOLD, 10f)
                alignmentX = Component.LEFT_ALIGNMENT
            })
            add(bodyLabel(value).apply {
                font = dankMonoFont(Font.PLAIN, 11f)
                alignmentX = Component.LEFT_ALIGNMENT
            })
        }
    }

    private fun buildDiagnosticTitle(index: Int, diagnostic: RunDiagnostic): String {
        return buildString {
            append("#").append(index).append(" ")
            append(diagnostic.strategy ?: diagnostic.bankId ?: diagnostic.status)
            if (diagnostic.isBest) {
                append(" | best")
            }
        }
    }

    private fun diagnosticTerminalSections(diagnostic: RunDiagnostic): List<Pair<String, String>> {
        val sections = mutableListOf<Pair<String, String>>()
        diagnostic.compileError?.takeIf { it.isNotBlank() }?.let {
            sections += "compile error" to it
        }
        if (diagnostic.visibleTestPassed == false) {
            testFailureOutput(
                stderr = diagnostic.visibleTestStderr,
                stdout = diagnostic.visibleTestStdout,
            )?.let {
                sections += labelWithReturnCode("visible test failure", diagnostic.visibleTestReturncode) to it
            }
        }
        if (diagnostic.hiddenTestPassed == false) {
            testFailureOutput(
                stderr = diagnostic.hiddenTestStderr,
                stdout = diagnostic.hiddenTestStdout,
            )?.let {
                sections += labelWithReturnCode("hidden test failure", diagnostic.hiddenTestReturncode) to it
            }
        }
        return sections
    }

    private fun testFailureOutput(stderr: String?, stdout: String?): String? {
        val parts = listOfNotNull(
            stderr?.takeIf { it.isNotBlank() },
            stdout?.takeIf { it.isNotBlank() },
        )
        return parts.takeIf { it.isNotEmpty() }?.joinToString("\n")
    }

    private fun terminalBlock(label: String, text: String): JComponent {
        val terminalText = truncateTerminalOutput(text)
        val area = JBTextArea(terminalText).apply {
            isEditable = false
            isFocusable = false
            isRequestFocusEnabled = false
            lineWrap = false
            wrapStyleWord = false
            foreground = Color(0xD7FFE3)
            caretColor = Color(0x050505)
            background = Color(0x05070A)
            border = JBUI.Borders.empty(10, 12, 10, 12)
            font = dankMonoFont(Font.PLAIN, 12f)
            isOpaque = true
            rows = max(3, min(10, terminalText.count { it == '\n' } + 1))
            columns = 1
        }
        return JPanel().apply {
            isOpaque = true
            background = Color(0x05070A)
            border = JBUI.Borders.empty(8, 8, 8, 8)
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            alignmentX = Component.LEFT_ALIGNMENT
            add(JBLabel(label.uppercase()).apply {
                foreground = Color(0x7C8796)
                font = dankMonoFont(Font.BOLD, 10f)
                alignmentX = Component.LEFT_ALIGNMENT
                border = JBUI.Borders.empty(0, 4, 6, 4)
            })
            add(JBScrollPane(area).apply {
                preferredSize = Dimension(100, min(220, max(92, area.rows * 22)))
                maximumSize = Dimension(Int.MAX_VALUE, 240)
                border = JBUI.Borders.empty()
                viewport.isOpaque = true
                viewport.background = Color(0x05070A)
                isOpaque = true
                background = Color(0x05070A)
                alignmentX = Component.LEFT_ALIGNMENT
                horizontalScrollBarPolicy = JBScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED
                verticalScrollBarPolicy = JBScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED
            })
        }
    }

    private fun diagnosticsSummary(diagnostics: List<RunDiagnostic>): String {
        val compileErrors = diagnostics.count { it.compileError?.isNotBlank() == true || it.status == "compile_failed" }
        val visibleFailures = diagnostics.count { it.visibleTestPassed == false || it.status == "visible_failed" }
        val hiddenFailures = diagnostics.count { it.hiddenTestPassed == false || it.status == "hidden_failed" }
        val fragments = mutableListOf<String>()
        if (compileErrors > 0) fragments += "$compileErrors compile error${if (compileErrors == 1) "" else "s"}"
        if (visibleFailures > 0) fragments += "$visibleFailures visible test failure${if (visibleFailures == 1) "" else "s"}"
        if (hiddenFailures > 0) fragments += "$hiddenFailures hidden test failure${if (hiddenFailures == 1) "" else "s"}"
        return fragments.ifEmpty { listOf("${diagnostics.size} compiled candidate${if (diagnostics.size == 1) "" else "s"}") }.joinToString(", ")
    }

    private fun labelWithReturnCode(label: String, returnCode: Int?): String {
        return if (returnCode == null) label else "$label (returncode=$returnCode)"
    }

    private fun truncateTerminalOutput(text: String): String {
        val limit = 12000
        return if (text.length <= limit) text else text.take(limit) + "\n[output truncated]"
    }

    private fun Boolean?.renderNullableBool(): String {
        return when (this) {
            true -> "pass"
            false -> "fail"
            null -> "-"
        }
    }

    private fun snapshot(): SearchGraphSnapshot {
        return stateStore.snapshot(
            showPruned = true,
            bestPathOnly = false,
        )
    }

    private fun titleLabel(text: String): JBLabel = JBLabel(text).apply {
        foreground = GraphPalette.textPrimary
        font = labelFont(Font.BOLD, 18f)
        alignmentX = Component.LEFT_ALIGNMENT
        horizontalAlignment = SwingConstants.LEFT
    }

    private fun bodyLabel(text: String): JBLabel = JBLabel(text).apply {
        foreground = GraphPalette.textSecondary
        font = labelFont(Font.PLAIN, 12f)
        alignmentX = Component.LEFT_ALIGNMENT
        horizontalAlignment = SwingConstants.LEFT
    }

    private fun metricLabel(text: String): JBLabel = JBLabel(text).apply {
        foreground = GraphPalette.textPrimary
        font = labelFont(Font.BOLD, 12f)
        horizontalAlignment = SwingConstants.LEFT
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
            alignmentX = Component.LEFT_ALIGNMENT
        }
    }

    private fun roundedCard(): JPanel {
        return object : JBPanel<JBPanel<*>>(BorderLayout()) {
            init {
                isOpaque = false
                border = JBUI.Borders.empty(12)
            }

            override fun paintComponent(graphics: Graphics) {
                val g2 = graphics.create() as Graphics2D
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
                val width = width - 1
                val height = height - 1
                g2.color = GraphPalette.shadow
                g2.fillRoundRect(2, 4, width - 3, height - 5, 24, 24)
                g2.paint = GradientPaint(0f, 0f, GraphPalette.card, width.toFloat(), height.toFloat(), GraphPalette.cardAlt)
                g2.fillRoundRect(0, 0, width, height - 2, 24, 24)
                g2.color = GraphPalette.border
                g2.drawRoundRect(0, 0, width, height - 2, 24, 24)
                g2.dispose()
                super.paintComponent(graphics)
            }
        }
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

    private fun strategyKey(policy: String): String = policy.lowercase()

    private fun showExpandedWindow() {
        val dialog = JDialog().apply {
            title = "Hyperreasoning Search Graph"
            isModal = false
            defaultCloseOperation = JDialog.DISPOSE_ON_CLOSE
            size = Dimension(1500, 960)
            minimumSize = Dimension(1100, 760)
            setLocationRelativeTo(null)
        }
        val expanded = SearchGraphPanel(project, mode)
        expanded.showSnapshot(
            snapshot = currentSnapshot,
            strategies = currentStrategies,
            strategyKey = currentStrategyKey,
            guidedPathEnabled = guidedPathEnabled,
            diagnosticsByBankId = diagnosticsByBankId,
        )
        dialog.contentPane.add(expanded)
        dialog.isVisible = true
        SwingUtilities.invokeLater {
            expanded.refitActiveView()
            SwingUtilities.invokeLater {
                expanded.refitActiveView()
            }
        }
    }

    private fun refitActiveView() {
        if (guidedPathEnabled) {
            timelinePanel.scrollToLatest()
        } else {
            canvas.fitToTree()
        }
    }

    private fun updateGuidedPathVisibility() {
        viewModeButton.text = if (guidedPathEnabled) "Tree View" else "Guided Path"
        viewModeButton.toolTipText = if (guidedPathEnabled) "Switch to Tree View" else "Switch to Guided Path"
        val targetKey = if (guidedPathEnabled) "timeline" else "tree"
        if (targetKey == activeViewKey) {
            viewCards.revalidate()
            viewCards.repaint()
            return
        }
        val direction = if (targetKey == "timeline") 1.0 else -1.0
        animatedViewSurface.animateSwap(direction) {
            viewCardLayout.show(viewCards, targetKey)
            viewCards.revalidate()
            viewCards.repaint()
        }
        activeViewKey = targetKey
    }
}

private class SpringSwapPanel(
    private val content: JComponent,
    backgroundColor: Color,
) : JPanel(BorderLayout()) {
    private var previousSnapshot: BufferedImage? = null
    private var currentSnapshot: BufferedImage? = null
    private var progress = 1.0
    private var direction = 1.0
    private val animationTimer: Timer = Timer(16, null)

    init {
        isOpaque = true
        background = backgroundColor
        add(content, BorderLayout.CENTER)
        animationTimer.addActionListener {
            progress = (progress + 0.11).coerceAtMost(1.0)
            if (progress >= 1.0) {
                previousSnapshot = null
                currentSnapshot = null
                animationTimer.stop()
            }
            repaint()
        }
    }

    fun animateSwap(direction: Double, swap: () -> Unit) {
        val oldImage = captureSnapshot()
        swap()
        doLayout()
        val newImage = captureSnapshot()
        if (oldImage == null || newImage == null) {
            previousSnapshot = null
            currentSnapshot = null
            progress = 1.0
            repaint()
            return
        }
        this.direction = direction
        previousSnapshot = oldImage
        currentSnapshot = newImage
        progress = 0.0
        animationTimer.restart()
    }

    override fun paintChildren(graphics: Graphics) {
        if (previousSnapshot == null || currentSnapshot == null || progress >= 1.0) {
            super.paintChildren(graphics)
            return
        }
        val g2 = graphics.create() as Graphics2D
        g2.color = background
        g2.fillRect(0, 0, width, height)
        val eased = springEase(progress)
        val oldAlpha = (1.0 - progress * 1.05).coerceIn(0.0, 1.0).toFloat()
        val newAlpha = (0.25 + progress * 0.9).coerceIn(0.0, 1.0).toFloat()
        val oldOffset = -direction * eased * width * 0.035
        val newOffset = direction * (1.0 - eased) * width * 0.09
        val newScale = 0.975 + 0.025 * eased
        val oldScale = 1.0 - 0.012 * progress
        previousSnapshot?.let { drawSnapshot(g2, it, oldOffset, oldScale, oldAlpha) }
        currentSnapshot?.let { drawSnapshot(g2, it, newOffset, newScale, newAlpha) }
        g2.dispose()
    }

    private fun drawSnapshot(
        g2: Graphics2D,
        image: BufferedImage,
        offsetX: Double,
        scale: Double,
        alpha: Float,
    ) {
        val copy = g2.create() as Graphics2D
        copy.composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, alpha.coerceIn(0f, 1f))
        val scaledWidth = image.width * scale
        val scaledHeight = image.height * scale
        val x = (width - scaledWidth) / 2.0 + offsetX
        val y = (height - scaledHeight) / 2.0
        copy.drawImage(image, x.toInt(), y.toInt(), scaledWidth.toInt(), scaledHeight.toInt(), null)
        copy.dispose()
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

    private fun springEase(t: Double): Double {
        val clamped = t.coerceIn(0.0, 1.0)
        return 1.0 - kotlin.math.exp(-8.0 * clamped) * kotlin.math.cos(10.0 * clamped)
    }
}

enum class SearchGraphPanelMode(
    val title: String,
    val emptySubtitle: String,
    val emptyDetails: String,
) {
    FULL_GRAPH(
        title = "Search Graph",
        emptySubtitle = "Run a strategy to visualize candidate generation, pruning, and the best path.",
        emptyDetails = "Select a node to inspect its metadata, score, and verification status.",
    ),
    FINAL_PATH(
        title = "Final Path",
        emptySubtitle = "Run a strategy to isolate the best branch only.",
        emptyDetails = "Select a node on the best path to inspect its metadata and verification status.",
    );

    fun subtitleFor(snapshot: SearchGraphSnapshot): String {
        return when (this) {
            FULL_GRAPH -> "Grey nodes are generated alternatives. Amber arrows show the executed trace and the current best path."
            FINAL_PATH -> "This view filters the graph to the current best path while preserving runtime order."
        }
    }
}

private class LegendChip(label: String, color: Color) : JPanel(BorderLayout(8, 0)) {
    init {
        isOpaque = false
        border = JBUI.Borders.empty(0, 0, 0, 0)
        add(object : JBPanel<JBPanel<*>>() {
            override fun getPreferredSize(): Dimension = Dimension(10, 10)

            override fun paintComponent(graphics: Graphics) {
                val g2 = graphics.create() as Graphics2D
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
                g2.color = color
                g2.fillOval(0, 0, width - 1, height - 1)
                g2.dispose()
            }
        }, BorderLayout.WEST)
        add(JBLabel(label).apply {
            foreground = GraphPalette.textSecondary
            font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.PLAIN, 12)).deriveFont(Font.PLAIN, 11f)
        }, BorderLayout.CENTER)
    }
}

private class StatusChip(title: String, valueLabel: JBLabel) : JPanel() {
    init {
        isOpaque = false
        layout = BoxLayout(this, BoxLayout.Y_AXIS)
        border = JBUI.Borders.empty(8, 10, 8, 10)
        add(JBLabel(title.uppercase()).apply {
            foreground = GraphPalette.textMuted
            font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 10f)
            alignmentX = Component.LEFT_ALIGNMENT
        })
        add(valueLabel.apply { alignmentX = Component.LEFT_ALIGNMENT })
    }

    override fun paintComponent(graphics: Graphics) {
        val g2 = graphics.create() as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g2.color = GraphPalette.surface
        g2.fillRoundRect(0, 0, width - 1, height - 1, 18, 18)
        g2.color = GraphPalette.border
        g2.drawRoundRect(0, 0, width - 1, height - 1, 18, 18)
        g2.dispose()
        super.paintComponent(graphics)
    }
}

private class GraphSplitPaneUI : BasicSplitPaneUI() {
    override fun createDefaultDivider(): BasicSplitPaneDivider {
        return object : BasicSplitPaneDivider(this) {
            init {
                border = JBUI.Borders.empty()
                background = GraphPalette.canvas
            }

            override fun paint(graphics: Graphics) {
                val g2 = graphics.create() as Graphics2D
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
                g2.color = GraphPalette.canvas
                g2.fillRect(0, 0, width, height)
                val gripWidth = if (orientation == JSplitPane.HORIZONTAL_SPLIT) 4 else width - 4
                val gripHeight = if (orientation == JSplitPane.HORIZONTAL_SPLIT) 42 else 4
                val gripX = if (orientation == JSplitPane.HORIZONTAL_SPLIT) (width - gripWidth) / 2 else 2
                val gripY = if (orientation == JSplitPane.HORIZONTAL_SPLIT) (height - gripHeight) / 2 else (height - gripHeight) / 2
                g2.paint = GradientPaint(
                    gripX.toFloat(),
                    gripY.toFloat(),
                    GraphPalette.border,
                    gripX.toFloat(),
                    (gripY + gripHeight).toFloat(),
                    GraphPalette.edgeMuted,
                )
                g2.fillRoundRect(gripX, gripY, gripWidth, gripHeight, 4, 4)
                g2.dispose()
            }
        }
    }
}

private class GraphActionButton(
    text: String,
    private val compact: Boolean = false,
    private val onSelect: () -> Unit,
) : JButton(text) {
    private var hoverProgress = 0.0
    private var pressProgress = 0.0
    private var shimmerPhase = Math.random() * 6.0
    private val animationTimer = Timer(16) {
        shimmerPhase += 0.045
        hoverProgress += (((if (model.isRollover) 1.0 else 0.0) - hoverProgress) * 0.22)
        pressProgress += (((if (model.isPressed) 1.0 else 0.0) - pressProgress) * 0.32)
        repaint()
    }

    init {
        isOpaque = false
        isContentAreaFilled = false
        isFocusPainted = false
        isBorderPainted = false
        cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
        foreground = GraphPalette.textPrimary
        font = (UIManager.getFont("Button.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, if (compact) 11f else 12f)
        border = JBUI.Borders.empty(if (compact) 6 else 10, if (compact) 10 else 12, if (compact) 6 else 10, if (compact) 10 else 12)
        addActionListener { onSelect() }
        animationTimer.start()
    }

    override fun paintComponent(graphics: Graphics) {
        val g2 = graphics.create() as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        val lift = hoverProgress * 2.0 - pressProgress * 1.5
        g2.translate(0.0, -lift)
        g2.color = GraphPalette.buttonShadow
        g2.fillRoundRect(1, 3, width - 3, height - 3, 18, 18)
        g2.color = when {
            !isEnabled -> GraphPalette.buttonDisabled
            model.isPressed -> GraphPalette.buttonPressed
            model.isRollover -> GraphPalette.buttonHover
            else -> GraphPalette.button
        }
        g2.fillRoundRect(0, 0, width - 1, height - 1, 18, 18)
        g2.paint = GradientPaint(
            ((shimmerPhase.sin01() * width) - width * 0.2).toFloat(),
            0f,
            GraphPalette.buttonSheen,
            ((shimmerPhase.sin01() * width) + width * 0.5).toFloat(),
            height.toFloat(),
            GraphPalette.buttonSheenSoft,
            true,
        )
        g2.fillRoundRect(0, 0, width - 1, height - 1, 18, 18)
        g2.color = GraphPalette.border
        g2.drawRoundRect(0, 0, width - 1, height - 1, 18, 18)
        g2.dispose()
        super.paintComponent(graphics)
    }
}

private class GraphStrategyButton(
    text: String,
    selected: Boolean,
    private val onSelect: () -> Unit,
) : JToggleButton(text, selected) {
    private var hoverProgress = 0.0
    private var selectionProgress = if (selected) 1.0 else 0.0
    private var shimmerPhase = Math.random() * 6.0
    private val animationTimer = Timer(16) {
        shimmerPhase += 0.04
        hoverProgress += (((if (model.isRollover) 1.0 else 0.0) - hoverProgress) * 0.2)
        selectionProgress += (((if (isSelected) 1.0 else 0.0) - selectionProgress) * 0.22)
        repaint()
    }

    init {
        isOpaque = false
        isContentAreaFilled = false
        isFocusPainted = false
        isBorderPainted = false
        isRolloverEnabled = true
        cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
        foreground = if (selected) GraphPalette.strategySelectedText else GraphPalette.textPrimary
        font = (UIManager.getFont("Button.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 12f)
        border = JBUI.Borders.empty(9, 12, 9, 12)
        addActionListener { onSelect() }
        animationTimer.start()
    }

    override fun paintComponent(graphics: Graphics) {
        foreground = if (isSelected) GraphPalette.strategySelectedText else GraphPalette.textPrimary
        val g2 = graphics.create() as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        val lift = hoverProgress * 2.2 + selectionProgress * 1.4
        g2.translate(0.0, -lift)
        g2.color = GraphPalette.buttonShadow
        g2.fillRoundRect(1, 4, width - 3, height - 3, 18, 18)
        g2.color = when {
            isSelected -> GraphPalette.strategySelected
            model.isPressed -> GraphPalette.buttonPressed
            model.isRollover -> GraphPalette.buttonHover
            else -> GraphPalette.button
        }
        g2.fillRoundRect(0, 0, width - 1, height - 1, 18, 18)
        g2.paint = GradientPaint(
            ((shimmerPhase.sin01() * width) - width * 0.18).toFloat(),
            0f,
            GraphPalette.buttonSheen,
            ((shimmerPhase.sin01() * width) + width * 0.48).toFloat(),
            height.toFloat(),
            GraphPalette.buttonSheenSoft,
            true,
        )
        g2.fillRoundRect(0, 0, width - 1, height - 1, 18, 18)
        g2.color = if (isSelected) GraphPalette.strategySelectedBorder else GraphPalette.border
        g2.drawRoundRect(0, 0, width - 1, height - 1, 18, 18)
        g2.dispose()
        super.paintComponent(graphics)
    }
}

private class GraphToggleButton(text: String, selected: Boolean, private val compact: Boolean = false) : JToggleButton(text, selected) {
    private var hoverProgress = 0.0
    private var selectionProgress = if (selected) 1.0 else 0.0
    private var shimmerPhase = Math.random() * 6.0
    private val animationTimer = Timer(16) {
        shimmerPhase += 0.04
        hoverProgress += (((if (model.isRollover) 1.0 else 0.0) - hoverProgress) * 0.2)
        selectionProgress += (((if (isSelected) 1.0 else 0.0) - selectionProgress) * 0.22)
        repaint()
    }

    init {
        isOpaque = false
        isContentAreaFilled = false
        isFocusPainted = false
        isBorderPainted = false
        cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
        foreground = GraphPalette.textPrimary
        font = (UIManager.getFont("Button.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, if (compact) 11f else 12f)
        border = JBUI.Borders.empty(if (compact) 6 else 10, if (compact) 10 else 12, if (compact) 6 else 10, if (compact) 10 else 12)
        animationTimer.start()
    }

    override fun paintComponent(graphics: Graphics) {
        val g2 = graphics.create() as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        val lift = hoverProgress * 1.8 + selectionProgress * 1.2
        g2.translate(0.0, -lift)
        g2.color = GraphPalette.buttonShadow
        g2.fillRoundRect(1, 4, width - 3, height - 3, 18, 18)
        g2.color = when {
            !isEnabled -> GraphPalette.buttonDisabled
            isSelected -> GraphPalette.toggleSelected
            model.isPressed -> GraphPalette.buttonPressed
            model.isRollover -> GraphPalette.buttonHover
            else -> GraphPalette.button
        }
        g2.fillRoundRect(0, 0, width - 1, height - 1, 18, 18)
        g2.paint = GradientPaint(
            ((shimmerPhase.sin01() * width) - width * 0.18).toFloat(),
            0f,
            GraphPalette.buttonSheen,
            ((shimmerPhase.sin01() * width) + width * 0.48).toFloat(),
            height.toFloat(),
            GraphPalette.buttonSheenSoft,
            true,
        )
        g2.fillRoundRect(0, 0, width - 1, height - 1, 18, 18)
        g2.color = if (isSelected) GraphPalette.toggleBorder else GraphPalette.border
        g2.drawRoundRect(0, 0, width - 1, height - 1, 18, 18)
        g2.dispose()
        super.paintComponent(graphics)
    }
}

private class SearchGraphCanvas(
    private val onNodeSelected: (String?) -> Unit,
) : JPanel() {
    private var snapshot: SearchGraphSnapshot? = null
    private var renderGraph: RenderGraph = RenderGraph(
        nodes = emptyMap(),
        edges = emptyList(),
        traceIndexByNodeId = emptyMap(),
        bestPathEdgeIds = emptySet(),
        terminalNodeId = null,
        bounds = GraphBounds(0.0, 0.0, 920.0, 560.0),
        worldWidth = 920.0,
        worldHeight = 560.0,
    )
    private var hoveredNodeId: String? = null
    private var selectedNodeId: String? = null
    private var zoom = 1.0
    private var dragStartPoint: Point? = null
    private var dragStartViewPosition: Point? = null
    private var dragMoved = false
    private var pulsePhase = 0.0
    private val animationTimer = Timer(16) {
        pulsePhase += 0.055
        renderGraph.nodes.values.forEach { node ->
            val stiffness = when {
                node.id == renderGraph.terminalNodeId -> 0.18
                node.state.isBestPath -> 0.15
                else -> 0.11
            }
            val damping = when {
                node.id == renderGraph.terminalNodeId -> 0.74
                node.state.isBestPath -> 0.78
                else -> 0.81
            }
            node.vx = (node.vx + (node.anchorX - node.x) * stiffness) * damping
            node.vy = (node.vy + (node.anchorY - node.y) * stiffness) * damping
            node.x += node.vx
            node.y += node.vy
            val targetScale = when {
                node.id == selectedNodeId -> 1.08
                node.id == hoveredNodeId -> 1.05
                node.id == renderGraph.terminalNodeId -> 1.1
                node.state.isBestPath -> 1.03
                else -> 1.0
            }
            node.scaleVelocity = (node.scaleVelocity + (targetScale - node.scale) * 0.18) * 0.72
            node.scale += node.scaleVelocity
        }
        repaint()
    }

    init {
        isOpaque = false
        preferredSize = Dimension(920, 560)
        cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
        toolTipText = ""
        val mouseHandler = object : MouseAdapter() {
            override fun mousePressed(event: MouseEvent) {
                dragStartPoint = event.point
                dragStartViewPosition = viewport()?.viewPosition
                dragMoved = false
                cursor = Cursor.getPredefinedCursor(Cursor.MOVE_CURSOR)
            }

            override fun mouseDragged(event: MouseEvent) {
                val origin = dragStartPoint ?: return
                val startView = dragStartViewPosition ?: return
                val viewport = viewport() ?: return
                val dx = event.x - origin.x
                val dy = event.y - origin.y
                dragMoved = dragMoved || hypot(dx.toDouble(), dy.toDouble()) >= 3.0
                val nextX = (startView.x - dx).coerceIn(0, max(0, width - viewport.extentSize.width))
                val nextY = (startView.y - dy).coerceIn(0, max(0, height - viewport.extentSize.height))
                viewport.viewPosition = Point(nextX, nextY)
            }

            override fun mouseReleased(event: MouseEvent) {
                if (!dragMoved) {
                    val hitNodeId = nodeAt(event.point)?.id
                    selectedNodeId = hitNodeId
                    onNodeSelected(hitNodeId)
                }
                dragStartPoint = null
                dragStartViewPosition = null
                dragMoved = false
                cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
                repaint()
            }

            override fun mouseMoved(event: MouseEvent) {
                hoveredNodeId = nodeAt(event.point)?.id
                repaint()
            }

            override fun mouseExited(event: MouseEvent) {
                hoveredNodeId = null
                repaint()
            }

            override fun mouseWheelMoved(event: MouseWheelEvent) {
                event.consume()
                adjustZoom(1.08.pow(-event.preciseWheelRotation), event.point)
            }

        }
        addMouseListener(mouseHandler)
        addMouseMotionListener(mouseHandler)
        addMouseWheelListener(mouseHandler)
        animationTimer.start()
    }

    override fun getToolTipText(event: MouseEvent): String? {
        val node = nodeAt(event.point) ?: return null
        return buildString {
            append("<html><b>").append(node.state.title).append("</b>")
            node.state.score?.let { append("<br/>score ").append(formatScore(it)) }
            append("<br/>status ").append(node.state.status.name.lowercase().replace('_', ' '))
            if (node.state.shortSummary.isNotBlank()) {
                append("<br/>").append(node.state.shortSummary)
            }
            append("</html>")
        }
    }

    fun setSnapshot(snapshot: SearchGraphSnapshot) {
        this.snapshot = snapshot
        selectedNodeId = snapshot.selectedNodeId ?: selectedNodeId
        renderGraph = buildRenderGraph(snapshot, renderGraph.nodes)
        updatePreferredSize()
        revalidate()
        repaint()
    }

    fun fitToTree() {
        val viewport = viewport() ?: return
        if (renderGraph.nodes.isEmpty()) {
            viewport.viewPosition = Point(0, 0)
            return
        }
        val contentWidth = (renderGraph.bounds.width + 220.0).coerceAtLeast(1.0)
        val contentHeight = (renderGraph.bounds.height + 220.0).coerceAtLeast(1.0)
        val availableWidth = viewport.extentSize.width.coerceAtLeast(200)
        val availableHeight = viewport.extentSize.height.coerceAtLeast(200)
        val targetZoom = min(
            availableWidth / contentWidth,
            availableHeight / contentHeight,
        ).coerceIn(0.18, 2.25)
        zoom = targetZoom
        updatePreferredSize()
        revalidate()
        centerViewportOn(renderGraph.bounds.centerX, renderGraph.bounds.centerY)
        repaint()
    }

    fun scrollToNode(nodeId: String) {
        val node = renderGraph.nodes[nodeId] ?: return
        centerViewportOn(node.x, node.y)
    }

    private fun centerViewportOn(worldX: Double, worldY: Double) {
        val viewport = viewport() ?: return
        val viewWidth = preferredSize.width.coerceAtLeast(width)
        val viewHeight = preferredSize.height.coerceAtLeast(height)
        val nextX = (worldX * zoom - viewport.extentSize.width / 2.0).toInt().coerceIn(0, max(0, viewWidth - viewport.extentSize.width))
        val nextY = (worldY * zoom - viewport.extentSize.height / 2.0).toInt().coerceIn(0, max(0, viewHeight - viewport.extentSize.height))
        viewport.viewPosition = Point(nextX, nextY)
    }

    override fun paintComponent(graphics: Graphics) {
        super.paintComponent(graphics)
        val g2 = graphics.create() as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g2.paint = GradientPaint(0f, 0f, GraphPalette.surface, width.toFloat(), height.toFloat(), GraphPalette.surfaceAlt)
        g2.fillRoundRect(0, 0, width, height, 22, 22)
        if (renderGraph.nodes.isEmpty()) {
            g2.color = GraphPalette.textSecondary
            g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.PLAIN, 12)).deriveFont(Font.PLAIN, 14f)
            val message = "No graph loaded."
            val metrics = g2.fontMetrics
            g2.drawString(message, (width - metrics.stringWidth(message)) / 2, height / 2)
            g2.dispose()
            return
        }
        drawBackdrop(g2)
        g2.scale(zoom, zoom)

        val selectedAncestors = selectedAncestorChain()
        renderGraph.edges.forEach { edge ->
            val from = renderGraph.nodes[edge.parentId] ?: return@forEach
            val to = renderGraph.nodes[edge.childId] ?: return@forEach
            drawTreeEdge(
                g2 = g2,
                from = from,
                to = to,
                muted = to.state.status == SearchGraphNodeStatus.PRUNED,
                onBestPath = edge.id in renderGraph.bestPathEdgeIds,
                onSelectedPath = edge.childId in selectedAncestors && edge.parentId in selectedAncestors,
            )
        }
        renderGraph.nodes.values.sortedBy { it.state.depth }.forEach { node ->
            drawNode(g2, node)
        }

        g2.dispose()
    }

    private fun drawBackdrop(g2: Graphics2D) {
        val shimmerOffset = ((pulsePhase * 42.0) % (width + 320.0)) - 160.0
        g2.paint = GradientPaint(
            shimmerOffset.toFloat(),
            0f,
            GraphPalette.canvasSheen,
            (shimmerOffset + 260.0).toFloat(),
            height.toFloat(),
            GraphPalette.canvasSheenSoft,
            true,
        )
        g2.fillRoundRect(0, 0, width, height, 22, 22)
        renderGraph.terminalNodeId
            ?.let(renderGraph.nodes::get)
            ?.let { terminal ->
                g2.color = GraphPalette.terminalBackdrop
                val radius = 180
                g2.fillOval(
                    (terminal.x * zoom - radius / 2).toInt(),
                    (terminal.y * zoom - radius / 2).toInt(),
                    radius,
                    radius,
                )
            }
    }

    private fun selectedAncestorChain(): Set<String> {
        val snapshot = snapshot ?: return emptySet()
        val parentById = snapshot.nodes.associateBy({ it.id }, { it.parentId })
        val result = linkedSetOf<String>()
        var current = selectedNodeId
        while (current != null) {
            result += current
            current = parentById[current]
        }
        return result
    }

    private fun buildRenderGraph(
        snapshot: SearchGraphSnapshot,
        previousNodes: Map<String, RenderNode>,
    ): RenderGraph {
        if (snapshot.nodes.isEmpty()) {
            return RenderGraph(
                nodes = emptyMap(),
                edges = emptyList(),
                traceIndexByNodeId = emptyMap(),
                bestPathEdgeIds = emptySet(),
                terminalNodeId = null,
                bounds = GraphBounds(0.0, 0.0, 920.0, 560.0),
                worldWidth = 920.0,
                worldHeight = 560.0,
            )
        }
        val childrenByParent = linkedMapOf<String?, MutableList<SearchGraphNodeState>>()
        snapshot.nodes
            .sortedWith(compareBy<SearchGraphNodeState> { it.depth }.thenBy { it.createdOrder })
            .forEach { node ->
                childrenByParent.getOrPut(node.parentId) { mutableListOf() }.add(node)
            }
        childrenByParent.values.forEach { siblings ->
            siblings.sortWith(compareBy<SearchGraphNodeState> { it.childIndex ?: Int.MAX_VALUE }.thenBy { it.createdOrder })
        }

        val leafIndex = mutableMapOf<String, Double>()
        var nextLeaf = 0.0

        fun assign(nodeId: String): Double {
            val children = childrenByParent[nodeId].orEmpty()
            val value = if (children.isEmpty()) {
                val current = nextLeaf
                nextLeaf += 1.0
                current
            } else {
                children.map { assign(it.id) }.average()
            }
            leafIndex[nodeId] = value
            return value
        }

        snapshot.nodes.filter { it.parentId == null }.forEach { assign(it.id) }
        val horizontalSpacing = 188.0
        val verticalSpacing = 122.0
        val minWorldWidth = 1280.0
        val minWorldHeight = 780.0
        val sidePadding = 180.0
        val topPadding = 126.0
        val nodeMetrics = snapshot.nodes.associateWith { node ->
            if (node.status == SearchGraphNodeStatus.ROOT) 108.0 to 44.0 else 150.0 to 72.0
        }
        val rawCenters = snapshot.nodes.associateWith { node ->
            (leafIndex[node.id] ?: 0.0) * horizontalSpacing to node.depth * verticalSpacing
        }
        val rawMinX = snapshot.nodes.minOf { node -> rawCenters.getValue(node).first - nodeMetrics.getValue(node).first / 2.0 }
        val rawMaxX = snapshot.nodes.maxOf { node -> rawCenters.getValue(node).first + nodeMetrics.getValue(node).first / 2.0 }
        val rawMinY = snapshot.nodes.minOf { node -> rawCenters.getValue(node).second - nodeMetrics.getValue(node).second / 2.0 }
        val rawMaxY = snapshot.nodes.maxOf { node -> rawCenters.getValue(node).second + nodeMetrics.getValue(node).second / 2.0 }
        val rawWidth = rawMaxX - rawMinX
        val rawHeight = rawMaxY - rawMinY
        val worldWidth = max(minWorldWidth, rawWidth + sidePadding * 2)
        val worldHeight = max(minWorldHeight, rawHeight + topPadding * 2)
        val shiftX = -rawMinX + (worldWidth - rawWidth) / 2.0
        val shiftY = -rawMinY + (worldHeight - rawHeight) / 2.0
        val renderNodes = linkedMapOf<String, RenderNode>()

        snapshot.nodes.forEach { node ->
            val previous = previousNodes[node.id]
            val isRoot = node.status == SearchGraphNodeStatus.ROOT
            val width = if (isRoot) 108.0 else 150.0
            val height = if (isRoot) 44.0 else 72.0
            val raw = rawCenters.getValue(node)
            val anchorX = raw.first + shiftX
            val anchorY = raw.second + shiftY
            renderNodes[node.id] = RenderNode(
                id = node.id,
                state = node,
                width = width,
                height = height,
                x = previous?.x ?: anchorX,
                y = previous?.y ?: anchorY,
                anchorX = anchorX,
                anchorY = anchorY,
                vx = previous?.vx ?: 0.0,
                vy = previous?.vy ?: 0.0,
                scale = previous?.scale ?: 1.0,
                scaleVelocity = previous?.scaleVelocity ?: 0.0,
            )
        }

        val renderEdges = snapshot.edges.map { edge ->
            RenderEdge(
                id = edge.id,
                parentId = edge.parentId,
                childId = edge.childId,
                actionLabel = edge.actionLabel,
            )
        }
        val traceIndexByNodeId = snapshot.bestPathIds.withIndex().associate { it.value to (it.index + 1) }
        val bestPathEdgeIds = snapshot.bestPathIds
            .zip(snapshot.bestPathIds.drop(1))
            .map { (from, to) -> "$from->$to" }
            .toSet()
        val bounds = GraphBounds(
            minX = renderNodes.values.minOf { it.anchorX - it.width / 2.0 },
            maxX = renderNodes.values.maxOf { it.anchorX + it.width / 2.0 },
            minY = renderNodes.values.minOf { it.anchorY - it.height / 2.0 },
            maxY = renderNodes.values.maxOf { it.anchorY + it.height / 2.0 },
        )
        return RenderGraph(
            nodes = renderNodes,
            edges = renderEdges,
            traceIndexByNodeId = traceIndexByNodeId,
            bestPathEdgeIds = bestPathEdgeIds,
            terminalNodeId = snapshot.successNodeId ?: snapshot.bestNodeId,
            bounds = bounds,
            worldWidth = worldWidth,
            worldHeight = worldHeight,
        )
    }

    private fun updatePreferredSize() {
        if (renderGraph.nodes.isEmpty()) {
            preferredSize = Dimension(920, 560)
            return
        }
        preferredSize = Dimension(
            (renderGraph.worldWidth * zoom).toInt().coerceAtLeast(980),
            (renderGraph.worldHeight * zoom).toInt().coerceAtLeast(620),
        )
    }

    private fun drawTreeEdge(
        g2: Graphics2D,
        from: RenderNode,
        to: RenderNode,
        muted: Boolean,
        onBestPath: Boolean,
        onSelectedPath: Boolean,
    ) {
        val fromY = from.y + from.height / 2.0
        val toY = to.y - to.height / 2.0
        val controlY = fromY + (toY - fromY) * 0.52
        val curve = Path2D.Double().apply {
            moveTo(from.x, fromY)
            curveTo(from.x, controlY, to.x, controlY - 8.0, to.x, toY)
        }
        if (onBestPath) {
            g2.stroke = BasicStroke(8.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND)
            g2.color = GraphPalette.sequenceGlow
            g2.draw(curve)
        }
        g2.stroke = BasicStroke(
            when {
                onSelectedPath -> 3.2f
                onBestPath -> 2.8f
                else -> 1.2f
            },
            BasicStroke.CAP_ROUND,
            BasicStroke.JOIN_ROUND,
        )
        g2.color = when {
            onSelectedPath -> GraphPalette.selection
            onBestPath -> GraphPalette.sequence
            muted -> GraphPalette.edgeMuted
            else -> GraphPalette.edge
        }
        g2.draw(curve)
        drawArrowHead(g2, Point(to.x.toInt(), (toY - 20).toInt()), Point(to.x.toInt(), toY.toInt()))
    }

    private fun drawNode(g2: Graphics2D, node: RenderNode) {
        val state = node.state
        val selected = state.id == selectedNodeId
        val hovered = state.id == hoveredNodeId
        val terminal = state.id == renderGraph.terminalNodeId
        if (state.status == SearchGraphNodeStatus.ROOT) {
            drawRootNode(g2, node, selected, terminal)
            return
        }

        val fill = statusFill(state)
        val animatedX = node.x
        if (state.status == SearchGraphNodeStatus.EXPANDING) {
            val pulseRadius = ((pulsePhase.sin01() * 8.0) + 6.0).toInt()
            g2.color = GraphPalette.expandingGlow
            g2.fillRoundRect(
                (animatedX - node.width / 2.0 - pulseRadius / 2.0).toInt(),
                ((node.y + hoverLift(node) * 0.6) - node.height / 2.0 - pulseRadius / 2.0).toInt(),
                (node.width + pulseRadius).toInt(),
                (node.height + pulseRadius).toInt(),
                24,
                24,
            )
        }
        if (terminal) {
            val pulseRadius = 22 + (pulsePhase.sin01() * 18.0).toInt()
            g2.color = if (state.status == SearchGraphNodeStatus.SUCCESS) GraphPalette.terminalHaloSuccess else GraphPalette.terminalHalo
            g2.fillRoundRect(
                (animatedX - node.width / 2.0 - pulseRadius / 2.0).toInt(),
                ((node.y + hoverLift(node)) - node.height / 2.0 - pulseRadius / 2.0).toInt(),
                (node.width + pulseRadius).toInt(),
                (node.height + pulseRadius).toInt(),
                34,
                34,
            )
        }
        if (state.status == SearchGraphNodeStatus.SUCCESS) {
            g2.color = GraphPalette.successGlow
            g2.fillRoundRect((animatedX - node.width / 2.0 - 6).toInt(), ((node.y + hoverLift(node)) - node.height / 2.0 - 6).toInt(), (node.width + 12).toInt(), (node.height + 12).toInt(), 24, 24)
        }
        val animatedY = node.y + hoverLift(node)
        val drawWidth = node.width * node.scale
        val drawHeight = node.height * node.scale
        g2.paint = GradientPaint(
            (animatedX - drawWidth / 2.0).toFloat(),
            (animatedY - drawHeight / 2.0).toFloat(),
            lift(fill, if (terminal) 0.16 else 0.09),
            (animatedX - drawWidth / 2.0).toFloat(),
            (animatedY + drawHeight / 2.0).toFloat(),
            GraphPalette.cardAlt,
        )
        g2.fillRoundRect((animatedX - drawWidth / 2.0).toInt(), (animatedY - drawHeight / 2.0).toInt(), drawWidth.toInt(), drawHeight.toInt(), 24, 24)
        g2.color = when {
            terminal -> GraphPalette.terminalRing
            selected -> GraphPalette.selection
            state.isBestPath -> GraphPalette.bestPath
            hovered -> GraphPalette.nodeHighlight
            else -> statusBorder(state)
        }
        g2.stroke = BasicStroke(if (selected || state.isBestPath || terminal) 2.8f else 1.2f)
        g2.drawRoundRect((animatedX - drawWidth / 2.0).toInt(), (animatedY - drawHeight / 2.0).toInt(), drawWidth.toInt(), drawHeight.toInt(), 24, 24)
        drawNodeText(g2, node, animatedX, animatedY, drawWidth, drawHeight)
        drawSlotBadge(g2, node, animatedX, animatedY, drawWidth, drawHeight)
        if (state.isBestPath) {
            drawBestPathBadge(g2, node, animatedX, animatedY, drawWidth, drawHeight)
        }
        if (terminal) {
            drawTerminalBadge(g2, state, animatedX, animatedY, drawWidth, drawHeight)
        }
    }

    private fun drawRootNode(g2: Graphics2D, node: RenderNode, selected: Boolean, terminal: Boolean) {
        val animatedY = node.y + hoverLift(node)
        val drawWidth = node.width * node.scale
        val drawHeight = node.height * node.scale
        g2.color = GraphPalette.rootGlow
        g2.fillRoundRect((node.x - drawWidth / 2.0 - 5).toInt(), (animatedY - drawHeight / 2.0 - 5).toInt(), (drawWidth + 10).toInt(), (drawHeight + 10).toInt(), 20, 20)
        g2.paint = GradientPaint(
            (node.x - drawWidth / 2.0).toFloat(),
            (animatedY - drawHeight / 2.0).toFloat(),
            GraphPalette.root,
            (node.x - drawWidth / 2.0).toFloat(),
            (animatedY + drawHeight / 2.0).toFloat(),
            GraphPalette.rootAlt,
        )
        g2.fillRoundRect((node.x - drawWidth / 2.0).toInt(), (animatedY - drawHeight / 2.0).toInt(), drawWidth.toInt(), drawHeight.toInt(), 18, 18)
        g2.color = when {
            terminal -> GraphPalette.terminalRing
            selected -> GraphPalette.selection
            else -> GraphPalette.border
        }
        g2.stroke = BasicStroke(if (selected || terminal) 2.3f else 1.4f)
        g2.drawRoundRect((node.x - drawWidth / 2.0).toInt(), (animatedY - drawHeight / 2.0).toInt(), drawWidth.toInt(), drawHeight.toInt(), 18, 18)
        val text = "ROOT"
        g2.color = GraphPalette.textPrimary
        g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 11f)
        val metrics = g2.fontMetrics
        g2.drawString(text, (node.x - metrics.stringWidth(text) / 2.0).toInt(), (animatedY + metrics.ascent / 2.0).toInt())
    }

    private fun drawNodeText(g2: Graphics2D, node: RenderNode, animatedX: Double, animatedY: Double, drawWidth: Double, drawHeight: Double) {
        val state = node.state
        val left = (animatedX - drawWidth / 2.0 + 12).toInt()
        val top = (animatedY - drawHeight / 2.0 + 18).toInt()
        g2.color = GraphPalette.textPrimary
        g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 11f)
        val title = ellipsize(state.title, 16)
        g2.drawString(title, left, top)
        g2.color = GraphPalette.textSecondary
        g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.PLAIN, 12)).deriveFont(Font.PLAIN, 10f)
        val statusText = buildString {
            append(state.status.name.lowercase().replace('_', ' '))
            state.score?.let { append(" • ").append(formatScore(it)) }
        }
        g2.drawString(ellipsize(statusText, 22), left, top + 15)
        if (state.shortSummary.isNotBlank()) {
            g2.drawString(ellipsize(state.shortSummary, 22), left, top + 29)
        }
    }

    private fun drawSlotBadge(g2: Graphics2D, node: RenderNode, animatedX: Double, animatedY: Double, drawWidth: Double, drawHeight: Double) {
        val slot = node.state.childIndex ?: return
        val badgeX = (animatedX - drawWidth / 2.0 - 8).toInt()
        val badgeY = (animatedY - drawHeight / 2.0 + 6).toInt()
        g2.color = GraphPalette.slotBadge
        g2.fillOval(badgeX, badgeY, 16, 16)
        g2.color = GraphPalette.slotBadgeBorder
        g2.drawOval(badgeX, badgeY, 16, 16)
        g2.color = GraphPalette.slotBadgeText
        g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 10f)
        val metrics = g2.fontMetrics
        g2.drawString(slot.toString(), badgeX + (16 - metrics.stringWidth(slot.toString())) / 2, badgeY + (16 + metrics.ascent - metrics.descent) / 2)
    }

    private fun drawBestPathBadge(g2: Graphics2D, node: RenderNode, animatedX: Double, animatedY: Double, drawWidth: Double, drawHeight: Double) {
        val step = renderGraph.traceIndexByNodeId[node.id] ?: return
        val badgeX = (animatedX + drawWidth / 2.0 - 14).toInt()
        val badgeY = (animatedY - drawHeight / 2.0 + 6).toInt()
        g2.color = GraphPalette.sequenceBadge
        g2.fillRoundRect(badgeX, badgeY, 18, 16, 10, 10)
        g2.color = GraphPalette.sequenceBadgeBorder
        g2.drawRoundRect(badgeX, badgeY, 18, 16, 10, 10)
        g2.color = GraphPalette.sequenceBadgeText
        g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 9f)
        val metrics = g2.fontMetrics
        g2.drawString(step.toString(), badgeX + (18 - metrics.stringWidth(step.toString())) / 2, badgeY + (16 + metrics.ascent - metrics.descent) / 2)
    }

    private fun drawTerminalBadge(
        g2: Graphics2D,
        state: SearchGraphNodeState,
        animatedX: Double,
        animatedY: Double,
        drawWidth: Double,
        drawHeight: Double,
    ) {
        val label = if (state.status == SearchGraphNodeStatus.SUCCESS) "SOLVED" else "FINAL"
        val badgeWidth = if (label == "SOLVED") 42 else 36
        val badgeHeight = 16
        val badgeX = (animatedX - badgeWidth / 2.0).toInt()
        val badgeY = (animatedY + drawHeight / 2.0 - 6).toInt()
        g2.color = if (state.status == SearchGraphNodeStatus.SUCCESS) GraphPalette.terminalBadge else GraphPalette.sequenceBadge
        g2.fillRoundRect(badgeX, badgeY, badgeWidth, badgeHeight, 12, 12)
        g2.color = GraphPalette.terminalRing
        g2.drawRoundRect(badgeX, badgeY, badgeWidth, badgeHeight, 12, 12)
        g2.color = GraphPalette.sequenceBadgeText
        g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 8.5f)
        val metrics = g2.fontMetrics
        g2.drawString(label, badgeX + (badgeWidth - metrics.stringWidth(label)) / 2, badgeY + (badgeHeight + metrics.ascent - metrics.descent) / 2)
    }

    private fun nodeAt(point: Point): RenderNode? {
        val worldX = point.x / zoom
        val worldY = point.y / zoom
        return renderGraph.nodes.values.lastOrNull { node ->
            val drawWidth = node.width * node.scale
            val drawHeight = node.height * node.scale
            worldX >= node.x - drawWidth / 2.0 &&
                worldX <= node.x + drawWidth / 2.0 &&
                worldY >= node.y - drawHeight / 2.0 &&
                worldY <= node.y + drawHeight / 2.0
        }
    }

    private fun adjustZoom(multiplier: Double, anchor: Point) {
        val viewport = viewport() ?: return
        val anchorInViewportX = anchor.x - viewport.viewPosition.x
        val anchorInViewportY = anchor.y - viewport.viewPosition.y
        val worldX = anchor.x / zoom
        val worldY = anchor.y / zoom
        zoom = (zoom * multiplier).coerceIn(0.25, 2.25)
        updatePreferredSize()
        revalidate()
        val nextViewX = (worldX * zoom - anchorInViewportX).toInt().coerceIn(0, max(0, preferredSize.width - viewport.extentSize.width))
        val nextViewY = (worldY * zoom - anchorInViewportY).toInt().coerceIn(0, max(0, preferredSize.height - viewport.extentSize.height))
        viewport.viewPosition = Point(nextViewX, nextViewY)
        repaint()
    }

    private fun viewport(): JViewport? {
        return SwingUtilities.getAncestorOfClass(JViewport::class.java, this) as? JViewport
    }

    private fun hoverLift(node: RenderNode): Double {
        return when {
            node.id == renderGraph.terminalNodeId -> kotlin.math.sin(pulsePhase * 0.9 + node.state.createdOrder * 0.25) * 4.0
            node.state.status == SearchGraphNodeStatus.EXPANDING -> kotlin.math.sin(pulsePhase + node.state.createdOrder * 0.35) * 3.5
            node.state.isBestPath -> kotlin.math.sin(pulsePhase * 0.7 + node.state.createdOrder * 0.2) * 1.8
            else -> 0.0
        }
    }

    private fun drawArrowHead(g2: Graphics2D, start: Point, tip: Point) {
        val dx = (tip.x - start.x).toDouble()
        val dy = (tip.y - start.y).toDouble()
        val distance = hypot(dx, dy).coerceAtLeast(1.0)
        val ux = dx / distance
        val uy = dy / distance
        val baseX = tip.x - ux * 12.0
        val baseY = tip.y - uy * 12.0
        val px = -uy
        val py = ux
        val polygon = Polygon(
            intArrayOf(
                tip.x,
                (baseX + px * 5.0).toInt(),
                (baseX - px * 5.0).toInt(),
            ),
            intArrayOf(
                tip.y,
                (baseY + py * 5.0).toInt(),
                (baseY - py * 5.0).toInt(),
            ),
            3,
        )
        g2.color = GraphPalette.sequence
        g2.fillPolygon(polygon)
    }

    private fun drawSequenceBadge(g2: Graphics2D, stepNumber: Int, centerX: Int, centerY: Int) {
        val label = stepNumber.toString()
        val width = 18 + label.length * 6
        val height = 16
        val x = centerX - width / 2
        val y = centerY - height / 2
        g2.color = GraphPalette.sequenceBadge
        g2.fillRoundRect(x, y, width, height, 10, 10)
        g2.color = GraphPalette.sequenceBadgeBorder
        g2.drawRoundRect(x, y, width, height, 10, 10)
        g2.color = GraphPalette.sequenceBadgeText
        g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 9f)
        val metrics = g2.fontMetrics
        g2.drawString(label, x + (width - metrics.stringWidth(label)) / 2, y + (height + metrics.ascent - metrics.descent) / 2)
    }

    private fun statusFill(state: SearchGraphNodeState): Color {
        return when (state.status) {
            SearchGraphNodeStatus.ROOT -> GraphPalette.root
            SearchGraphNodeStatus.ACTIVE -> GraphPalette.active
            SearchGraphNodeStatus.EXPANDING -> GraphPalette.expanding
            SearchGraphNodeStatus.PRUNED -> GraphPalette.pruned
            SearchGraphNodeStatus.SUCCESS -> GraphPalette.success
            SearchGraphNodeStatus.FAILED_COMPILE -> GraphPalette.failedCompile
            SearchGraphNodeStatus.FAILED_TEST -> GraphPalette.failedTest
            SearchGraphNodeStatus.FAILED_RUNTIME -> GraphPalette.failedRuntime
            SearchGraphNodeStatus.IDLE -> GraphPalette.node
        }
    }

    private fun statusBorder(state: SearchGraphNodeState): Color {
        return when (state.status) {
            SearchGraphNodeStatus.PRUNED -> GraphPalette.prunedBorder
            SearchGraphNodeStatus.SUCCESS -> GraphPalette.successBorder
            SearchGraphNodeStatus.FAILED_COMPILE, SearchGraphNodeStatus.FAILED_TEST, SearchGraphNodeStatus.FAILED_RUNTIME -> GraphPalette.failureBorder
            SearchGraphNodeStatus.EXPANDING -> GraphPalette.expandingBorder
            else -> GraphPalette.border
        }
    }

    private data class RenderGraph(
        val nodes: Map<String, RenderNode>,
        val edges: List<RenderEdge>,
        val traceIndexByNodeId: Map<String, Int>,
        val bestPathEdgeIds: Set<String>,
        val terminalNodeId: String?,
        val bounds: GraphBounds,
        val worldWidth: Double,
        val worldHeight: Double,
    )

    private data class GraphBounds(
        val minX: Double,
        val maxX: Double,
        val minY: Double,
        val maxY: Double,
    ) {
        val width: Double get() = maxX - minX
        val height: Double get() = maxY - minY
        val centerX: Double get() = (minX + maxX) / 2.0
        val centerY: Double get() = (minY + maxY) / 2.0
    }

    private data class RenderEdge(
        val id: String,
        val parentId: String,
        val childId: String,
        val actionLabel: String?,
    )

    private data class RenderNode(
        val id: String,
        val state: SearchGraphNodeState,
        val width: Double,
        val height: Double,
        var x: Double,
        var y: Double,
        var anchorX: Double,
        var anchorY: Double,
        var vx: Double,
        var vy: Double,
        var scale: Double,
        var scaleVelocity: Double,
    )
}

private class GuidedPathTimelinePanel(
    private val onNodeSelected: (String?) -> Unit,
) : JPanel() {
    private var snapshot: SearchGraphSnapshot? = null
    private var steps: List<SearchGraphDecisionStep> = emptyList()
    private var hoveredStepIndex: Int? = null
    private var selectedStepIndex: Int? = null
    private var sprites: MutableList<TimelineStepSprite> = mutableListOf()
    private var draggingIndex: Int? = null
    private var dragOffset = Point()
    private var dragMoved = false
    private var pulsePhase = 0.0
    private val animationTimer = Timer(16) {
        pulsePhase += 0.12
        sprites.forEachIndexed { index, sprite ->
            if (draggingIndex == index) return@forEachIndexed
            sprite.vx = (sprite.vx + (sprite.anchorX - sprite.x) * 0.08) * 0.82
            sprite.vy = (sprite.vy + (sprite.anchorY - sprite.y) * 0.08) * 0.82
            sprite.x += sprite.vx
            sprite.y += sprite.vy
        }
        repaint()
    }

    init {
        isOpaque = false
        preferredSize = Dimension(1120, 208)
        cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
        toolTipText = ""
        val mouseHandler = object : MouseAdapter() {
            override fun mousePressed(event: MouseEvent) {
                val index = stepIndexAt(event.point) ?: return
                draggingIndex = index
                val rect = spriteRect(index)
                dragOffset = Point(event.x - rect.x, event.y - rect.y)
                dragMoved = false
                cursor = Cursor.getPredefinedCursor(Cursor.MOVE_CURSOR)
            }

            override fun mouseDragged(event: MouseEvent) {
                val index = draggingIndex ?: return
                val sprite = sprites.getOrNull(index) ?: return
                sprite.x = (event.x - dragOffset.x).toDouble()
                sprite.y = (event.y - dragOffset.y).toDouble()
                sprite.vx = 0.0
                sprite.vy = 0.0
                dragMoved = true
                repaint()
            }

            override fun mouseReleased(event: MouseEvent) {
                val index = stepIndexAt(event.point)
                if (!dragMoved) {
                    selectedStepIndex = index
                    onNodeSelected(index?.let { steps[it].nodeId })
                }
                draggingIndex = null
                dragMoved = false
                cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
                repaint()
            }

            override fun mouseMoved(event: MouseEvent) {
                hoveredStepIndex = stepIndexAt(event.point)
                repaint()
            }

            override fun mouseExited(event: MouseEvent) {
                hoveredStepIndex = null
                repaint()
            }
        }
        addMouseListener(mouseHandler)
        addMouseMotionListener(mouseHandler)
        animationTimer.start()
    }

    override fun getToolTipText(event: MouseEvent): String? {
        val index = stepIndexAt(event.point) ?: return null
        val step = steps[index]
        return buildString {
            append("<html><b>").append(step.title).append("</b>")
            append("<br/>step ").append(step.order)
            step.transitionLabel?.takeIf { it.isNotBlank() }?.let { append("<br/>").append(it) }
            step.score?.let { append("<br/>score ").append(formatScore(it)) }
            append("</html>")
        }
    }

    fun setSnapshot(snapshot: SearchGraphSnapshot) {
        this.snapshot = snapshot
        steps = snapshot.decisionTrail
        if (selectedStepIndex != null && selectedStepIndex!! >= steps.size) {
            selectedStepIndex = null
        }
        rebuildSprites()
        updatePreferredSize()
        revalidate()
        repaint()
    }

    fun scrollToLatest() {
        if (steps.isEmpty()) return
        val viewport = viewport() ?: return
        val lastRect = spriteRect(steps.lastIndex)
        viewport.viewPosition = Point(
            max(0, lastRect.x + lastRect.width - viewport.extentSize.width + 32),
            0,
        )
    }

    override fun paintComponent(graphics: Graphics) {
        super.paintComponent(graphics)
        val g2 = graphics.create() as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g2.paint = GradientPaint(0f, 0f, GraphPalette.surface, width.toFloat(), height.toFloat(), GraphPalette.surfaceAlt)
        g2.fillRoundRect(0, 0, width, height, 22, 22)
        if (steps.isEmpty()) {
            g2.color = GraphPalette.textSecondary
            g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.PLAIN, 12)).deriveFont(Font.PLAIN, 13f)
            val message = "No guided decisions yet."
            val metrics = g2.fontMetrics
            g2.drawString(message, (width - metrics.stringWidth(message)) / 2, height / 2)
            g2.dispose()
            return
        }

        steps.zipWithNext().forEachIndexed { index, (fromStep, toStep) ->
            drawConnector(g2, index, fromStep, toStep)
        }
        steps.forEachIndexed { index, step ->
            drawStep(g2, index, step)
        }
        g2.dispose()
    }

    private fun drawConnector(g2: Graphics2D, index: Int, fromStep: SearchGraphDecisionStep, toStep: SearchGraphDecisionStep) {
        val fromRect = spriteRect(index)
        val toRect = spriteRect(index + 1)
        val startX = fromRect.x + fromRect.width
        val endX = toRect.x
        val centerY = fromRect.y + fromRect.height / 2
        val path = Path2D.Double().apply {
            moveTo(startX.toDouble(), centerY.toDouble())
            curveTo(
                startX + 42.0,
                centerY.toDouble(),
                endX - 42.0,
                centerY.toDouble(),
                endX.toDouble(),
                centerY.toDouble(),
            )
        }
        g2.stroke = BasicStroke(2.4f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND)
        g2.color = if (toStep.isBestPath) GraphPalette.sequence else GraphPalette.edge
        g2.draw(path)
        val pulseT = ((pulsePhase + index * 0.35) % 6.0) / 6.0
        val orbX = (startX + (endX - startX) * pulseT).toInt()
        g2.color = if (toStep.isBestPath) GraphPalette.sequenceGlow else GraphPalette.nodeHighlight
        g2.fillOval(orbX - 4, centerY - 4, 8, 8)
        drawArrowHead(g2, Point(endX - 18, centerY), Point(endX - 4, centerY))
    }

    private fun drawArrowHead(g2: Graphics2D, start: Point, tip: Point) {
        val dx = (tip.x - start.x).toDouble()
        val dy = (tip.y - start.y).toDouble()
        val distance = hypot(dx, dy).coerceAtLeast(1.0)
        val ux = dx / distance
        val uy = dy / distance
        val baseX = tip.x - ux * 10.0
        val baseY = tip.y - uy * 10.0
        val px = -uy
        val py = ux
        val polygon = Polygon(
            intArrayOf(
                tip.x,
                (baseX + px * 4.0).toInt(),
                (baseX - px * 4.0).toInt(),
            ),
            intArrayOf(
                tip.y,
                (baseY + py * 4.0).toInt(),
                (baseY - py * 4.0).toInt(),
            ),
            3,
        )
        g2.color = GraphPalette.sequence
        g2.fillPolygon(polygon)
    }

    private fun drawStep(g2: Graphics2D, index: Int, step: SearchGraphDecisionStep) {
        val rect = spriteRect(index)
        val selected = selectedStepIndex == index
        val hovered = hoveredStepIndex == index
        val terminal = index == steps.lastIndex
        if (selected || hovered || terminal) {
            g2.color = GraphPalette.selection
            g2.fillRoundRect(rect.x - 3, rect.y - 3, rect.width + 6, rect.height + 6, 22, 22)
        }
        if (terminal) {
            g2.color = GraphPalette.terminalHalo
            g2.fillRoundRect(rect.x - 8, rect.y - 8, rect.width + 16, rect.height + 16, 26, 26)
        }
        g2.paint = GradientPaint(
            rect.x.toFloat(),
            rect.y.toFloat(),
            lift(statusFill(step.status), if (terminal) 0.16 else 0.08),
            rect.x.toFloat(),
            (rect.y + rect.height).toFloat(),
            GraphPalette.cardAlt,
        )
        g2.fillRoundRect(rect.x, rect.y, rect.width, rect.height, 18, 18)
        g2.color = when {
            terminal -> GraphPalette.terminalRing
            selected -> GraphPalette.selection
            step.isBestPath -> GraphPalette.bestPath
            else -> statusBorder(step.status)
        }
        g2.stroke = BasicStroke(if (selected || step.isBestPath || terminal) 2.4f else 1.1f)
        g2.drawRoundRect(rect.x, rect.y, rect.width, rect.height, 18, 18)

        g2.color = GraphPalette.sequenceBadge
        g2.fillRoundRect(rect.x + 8, rect.y + 8, 26, 16, 10, 10)
        g2.color = GraphPalette.sequenceBadgeBorder
        g2.drawRoundRect(rect.x + 8, rect.y + 8, 26, 16, 10, 10)
        g2.color = GraphPalette.sequenceBadgeText
        g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 9f)
        val badgeMetrics = g2.fontMetrics
        val orderText = step.order.toString()
        g2.drawString(orderText, rect.x + 8 + (26 - badgeMetrics.stringWidth(orderText)) / 2, rect.y + 8 + (16 + badgeMetrics.ascent - badgeMetrics.descent) / 2)

        g2.color = GraphPalette.textPrimary
        g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 11f)
        g2.drawString(ellipsize(step.title, 18), rect.x + 10, rect.y + 42)
        g2.color = GraphPalette.textSecondary
        g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.PLAIN, 12)).deriveFont(Font.PLAIN, 10f)
        val lineTwo = buildString {
            append(step.status.name.lowercase().replace('_', ' '))
            step.childIndex?.let { append(" • child ").append(it) }
        }
        g2.drawString(ellipsize(lineTwo, 24), rect.x + 10, rect.y + 58)
        step.transitionLabel?.takeIf { it.isNotBlank() }?.let {
            g2.drawString(ellipsize("via ${it}", 24), rect.x + 10, rect.y + 74)
        }
        step.score?.let {
            g2.drawString("score ${formatScore(it)}", rect.x + 10, rect.y + 90)
        }
        if (terminal) {
            val label = if (step.status == SearchGraphNodeStatus.SUCCESS) "SOLVED" else "FINAL"
            g2.color = if (step.status == SearchGraphNodeStatus.SUCCESS) GraphPalette.terminalBadge else GraphPalette.sequenceBadge
            g2.fillRoundRect(rect.x + rect.width - 48, rect.y + rect.height - 22, 40, 14, 10, 10)
            g2.color = GraphPalette.terminalRing
            g2.drawRoundRect(rect.x + rect.width - 48, rect.y + rect.height - 22, 40, 14, 10, 10)
            g2.color = GraphPalette.sequenceBadgeText
            g2.font = (UIManager.getFont("Label.font") ?: Font(Font.SANS_SERIF, Font.BOLD, 12)).deriveFont(Font.BOLD, 8.5f)
            val metrics = g2.fontMetrics
            g2.drawString(label, rect.x + rect.width - 48 + (40 - metrics.stringWidth(label)) / 2, rect.y + rect.height - 22 + (14 + metrics.ascent - metrics.descent) / 2)
        }
    }

    private fun spriteRect(index: Int): java.awt.Rectangle {
        val sprite = sprites[index]
        return java.awt.Rectangle(sprite.x.toInt(), sprite.y.toInt(), sprite.width, sprite.height)
    }

    private fun updatePreferredSize() {
        val width = if (sprites.isEmpty()) {
            1120
        } else {
            max(1120, (sprites.maxOf { it.anchorX + it.width } + 96).toInt())
        }
        preferredSize = Dimension(width, 208)
    }

    private fun stepIndexAt(point: Point): Int? {
        return steps.indices.firstOrNull { spriteRect(it).contains(point) }
    }

    private fun viewport(): JViewport? {
        return SwingUtilities.getAncestorOfClass(JViewport::class.java, this) as? JViewport
    }

    private fun statusFill(status: SearchGraphNodeStatus): Color {
        return when (status) {
            SearchGraphNodeStatus.ROOT -> GraphPalette.root
            SearchGraphNodeStatus.ACTIVE -> GraphPalette.active
            SearchGraphNodeStatus.EXPANDING -> GraphPalette.expanding
            SearchGraphNodeStatus.PRUNED -> GraphPalette.pruned
            SearchGraphNodeStatus.SUCCESS -> GraphPalette.success
            SearchGraphNodeStatus.FAILED_COMPILE -> GraphPalette.failedCompile
            SearchGraphNodeStatus.FAILED_TEST -> GraphPalette.failedTest
            SearchGraphNodeStatus.FAILED_RUNTIME -> GraphPalette.failedRuntime
            SearchGraphNodeStatus.IDLE -> GraphPalette.node
        }
    }

    private fun statusBorder(status: SearchGraphNodeStatus): Color {
        return when (status) {
            SearchGraphNodeStatus.PRUNED -> GraphPalette.prunedBorder
            SearchGraphNodeStatus.SUCCESS -> GraphPalette.successBorder
            SearchGraphNodeStatus.FAILED_COMPILE, SearchGraphNodeStatus.FAILED_TEST, SearchGraphNodeStatus.FAILED_RUNTIME -> GraphPalette.failureBorder
            SearchGraphNodeStatus.EXPANDING -> GraphPalette.expandingBorder
            else -> GraphPalette.border
        }
    }

    private fun rebuildSprites() {
        val previousByOrder = sprites.associateBy { it.order }
        val rebuilt = mutableListOf<TimelineStepSprite>()
        val stepWidth = 188.0
        val stepGap = 54.0
        val contentWidth = if (steps.isEmpty()) 0.0 else steps.size * stepWidth + (steps.size - 1) * stepGap
        val xOffset = max(0.0, (1120.0 - contentWidth) / 2.0)
        steps.forEachIndexed { index, step ->
            val anchorX = xOffset + 24.0 + index * (stepWidth + stepGap)
            val anchorY = 46.0 + if (index % 2 == 0) 0.0 else 12.0
            val previous = previousByOrder[step.order]
            rebuilt += TimelineStepSprite(
                order = step.order,
                x = previous?.x ?: anchorX,
                y = previous?.y ?: anchorY,
                anchorX = anchorX,
                anchorY = anchorY,
                width = stepWidth.toInt(),
                height = 102,
                vx = previous?.vx ?: 0.0,
                vy = previous?.vy ?: 0.0,
            )
        }
        sprites = rebuilt
    }

    private data class TimelineStepSprite(
        val order: Int,
        var x: Double,
        var y: Double,
        var anchorX: Double,
        var anchorY: Double,
        val width: Int,
        val height: Int,
        var vx: Double,
        var vy: Double,
    )
}

private object GraphPalette {
    val canvas = JBColor(Color(0xF4F5F7), Color(0x111315))
    val canvasSheen = JBColor(Color(0x14FFFFFF, true), Color(0x16FFFFFF, true))
    val canvasSheenSoft = JBColor(Color(0x00FFFFFF, true), Color(0x00FFFFFF, true))
    val card = JBColor(Color(0xFCFCFD), Color(0x181B1E))
    val cardAlt = JBColor(Color(0xF1F3F6), Color(0x15181B))
    val surface = JBColor(Color(0xF7F8FA), Color(0x14171A))
    val surfaceAlt = JBColor(Color(0xEEF1F4), Color(0x111417))
    val border = JBColor(Color(0xD9DEE5), Color(0x2C333A))
    val textPrimary = JBColor(Color(0x111315), Color(0xF4F5F6))
    val textSecondary = JBColor(Color(0x616873), Color(0xA0A8B2))
    val textMuted = JBColor(Color(0x8A939D), Color(0x7A858F))
    val shadow = JBColor(Color(0x12000000, true), Color(0x24000000, true))
    val edge = JBColor(Color(0xA9B4C0), Color(0x55606B))
    val edgeMuted = JBColor(Color(0x8B95A1), Color(0x39424B))
    val active = JBColor(Color(0xDCE6F7), Color(0x2A4157))
    val expanding = JBColor(Color(0xFFF1D8), Color(0x5A3F17))
    val expandingGlow = JBColor(Color(0xFFE3B0), Color(0x6B4715))
    val expandingBorder = JBColor(Color(0xD88916), Color(0xFFC978))
    val bestPath = JBColor(Color(0xD97706), Color(0xFDBA74))
    val sequence = JBColor(Color(0xFFB34D), Color(0xFFC978))
    val sequenceGlow = JBColor(Color(0xFFE7BE), Color(0x6B4715))
    val sequenceBadge = JBColor(Color(0xFFF4E1), Color(0x4B3620))
    val sequenceBadgeBorder = JBColor(Color(0xD97706), Color(0xFFC978))
    val sequenceBadgeText = JBColor(Color(0x7A3D00), Color(0xFFF4E1))
    val success = JBColor(Color(0xDFF3E4), Color(0x244E36))
    val successGlow = JBColor(Color(0xD6F4DE), Color(0x215039))
    val successBorder = JBColor(Color(0x3D8A58), Color(0x63C684))
    val terminalHalo = JBColor(Color(0x5BE7B56B, true), Color(0x3CD6A24A, true))
    val terminalHaloSuccess = JBColor(Color(0x63AAF3B1, true), Color(0x3F44C77B, true))
    val terminalRing = JBColor(Color(0xF59E0B), Color(0xFCD34D))
    val terminalBadge = JBColor(Color(0xEAF8EE), Color(0x1D5638))
    val terminalBackdrop = JBColor(Color(0x33F5B34D, true), Color(0x1ECC9A3B, true))
    val failedCompile = JBColor(Color(0xFDE2E1), Color(0x5E2A2A))
    val failedTest = JBColor(Color(0xFFE6CC), Color(0x61381C))
    val failedRuntime = JBColor(Color(0xF6DCE8), Color(0x5D2B42))
    val failureBorder = JBColor(Color(0xD25C5C), Color(0xFF9989))
    val pruned = JBColor(Color(0xD7DDE5), Color(0x30363D))
    val prunedBorder = JBColor(Color(0x9AA4B0), Color(0x616C77))
    val root = JBColor(Color(0xB0B8C1), Color(0x49515A))
    val rootAlt = JBColor(Color(0xC4CBD3), Color(0x5D6670))
    val rootGlow = JBColor(Color(0xE8EDF2), Color(0x313941))
    val node = JBColor(Color(0xC7CED6), Color(0x3A434B))
    val nodeHighlight = JBColor(Color(0xE9EDF2), Color(0x67727D))
    val selection = JBColor(Color(0x3B82F6), Color(0x60A5FA))
    val button = JBColor(Color(0xF3F5F8), Color(0x20252A))
    val buttonHover = JBColor(Color(0xE8ECF1), Color(0x2A3036))
    val buttonPressed = JBColor(Color(0xDDE4EB), Color(0x333B43))
    val buttonDisabled = JBColor(Color(0xECEFF3), Color(0x1C2025))
    val buttonShadow = JBColor(Color(0x16000000, true), Color(0x28000000, true))
    val buttonSheen = JBColor(Color(0x28FFFFFF, true), Color(0x1DFFFFFF, true))
    val buttonSheenSoft = JBColor(Color(0x00FFFFFF, true), Color(0x00FFFFFF, true))
    val strategySelected = JBColor(Color(0xFFFFFF), Color(0xF4F5F6))
    val strategySelectedBorder = JBColor(Color(0xD9DEE5), Color(0xF4F5F6))
    val strategySelectedText = JBColor(Color(0x111315), Color(0x111315))
    val toggleSelected = JBColor(Color(0xE4EAF4), Color(0x24303B))
    val toggleBorder = JBColor(Color(0x8EA4BC), Color(0xAAC4DE))
    val slotBadge = JBColor(Color(0xE7EDF4), Color(0x22303C))
    val slotBadgeBorder = JBColor(Color(0x8FA2B6), Color(0x9FB4C9))
    val slotBadgeText = JBColor(Color(0x213140), Color(0xE7EDF4))
}

private fun formatScore(value: Double): String = String.format("%.2f", value)

private fun ellipsize(text: String, maxChars: Int): String {
    if (text.length <= maxChars) {
        return text
    }
    return text.take(max(1, maxChars - 1)) + "…"
}

private fun Double.sin01(): Double {
    return (kotlin.math.sin(this) + 1.0) / 2.0
}

private fun lift(color: Color, amount: Double): Color {
    fun channel(value: Int): Int = (value + ((255 - value) * amount)).toInt().coerceIn(0, 255)
    return Color(channel(color.red), channel(color.green), channel(color.blue), color.alpha)
}
