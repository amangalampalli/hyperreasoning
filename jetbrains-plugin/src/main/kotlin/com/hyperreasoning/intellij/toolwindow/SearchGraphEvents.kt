package com.hyperreasoning.intellij.toolwindow

enum class SearchGraphNodeStatus {
    ROOT,
    ACTIVE,
    EXPANDING,
    PRUNED,
    SUCCESS,
    FAILED_COMPILE,
    FAILED_TEST,
    FAILED_RUNTIME,
    IDLE,
}

enum class SearchGraphLifecycleStatus {
    IDLE,
    WAITING,
    RUNNING,
    PAUSED,
    FINISHED,
    FAILED,
}

data class SearchGraphNodePayload(
    val id: String,
    val parentId: String? = null,
    val depth: Int = 0,
    val title: String,
    val shortSummary: String = "",
    val dslSummary: String = "",
    val patchSummary: String = "",
    val rationaleSummary: String = "",
    val childIndex: Int? = null,
    val childCount: Int? = null,
    val status: SearchGraphNodeStatus = SearchGraphNodeStatus.IDLE,
    val createdOrder: Int = 0,
    val createdAtLabel: String? = null,
    val rawMetadataJson: String? = null,
)

data class SearchGraphScorePayload(
    val id: String,
    val score: Double? = null,
    val rank: Int? = null,
    val qValue: Double? = null,
    val heuristicScore: Double? = null,
)

sealed interface SearchGraphEvent {
    val runId: String
}

data class SearchStartedEvent(
    override val runId: String,
    val title: String,
    val subtitle: String,
) : SearchGraphEvent

data class SearchResetEvent(
    override val runId: String,
) : SearchGraphEvent

data class NodeCreatedEvent(
    override val runId: String,
    val node: SearchGraphNodePayload,
) : SearchGraphEvent

data class NodeUpdatedEvent(
    override val runId: String,
    val nodeId: String,
    val title: String? = null,
    val shortSummary: String? = null,
    val patchSummary: String? = null,
    val rationaleSummary: String? = null,
    val childIndex: Int? = null,
    val childCount: Int? = null,
    val rawMetadataJson: String? = null,
) : SearchGraphEvent

data class EdgeCreatedEvent(
    override val runId: String,
    val parentId: String,
    val childId: String,
    val actionLabel: String? = null,
) : SearchGraphEvent

data class NodeScoredEvent(
    override val runId: String,
    val score: SearchGraphScorePayload,
) : SearchGraphEvent

data class NodePrunedEvent(
    override val runId: String,
    val nodeId: String,
    val reason: String,
) : SearchGraphEvent

data class NodeStatusChangedEvent(
    override val runId: String,
    val nodeId: String,
    val status: SearchGraphNodeStatus,
    val terminalSummary: String? = null,
    val compileStatus: String? = null,
    val testStatus: String? = null,
    val runtimeStatus: String? = null,
) : SearchGraphEvent

data class BestPathUpdatedEvent(
    override val runId: String,
    val nodeIds: List<String>,
) : SearchGraphEvent

data class SearchFinishedEvent(
    override val runId: String,
    val terminalNodeId: String? = null,
    val success: Boolean = true,
    val summary: String,
) : SearchGraphEvent

data class SearchGraphNodeState(
    val id: String,
    val parentId: String?,
    val depth: Int,
    val title: String,
    val shortSummary: String,
    val dslSummary: String,
    val patchSummary: String,
    val rationaleSummary: String,
    val childIndex: Int?,
    val childCount: Int?,
    val status: SearchGraphNodeStatus,
    val createdOrder: Int,
    val createdAtLabel: String?,
    val score: Double?,
    val rank: Int?,
    val qValue: Double?,
    val heuristicScore: Double?,
    val isBestPath: Boolean,
    val terminalSummary: String?,
    val compileStatus: String?,
    val testStatus: String?,
    val runtimeStatus: String?,
    val rawMetadataJson: String?,
)

data class SearchGraphEdgeState(
    val id: String,
    val parentId: String,
    val childId: String,
    val actionLabel: String?,
)

data class SearchGraphDecisionStep(
    val order: Int,
    val nodeId: String,
    val title: String,
    val status: SearchGraphNodeStatus,
    val transitionLabel: String?,
    val childIndex: Int?,
    val score: Double?,
    val isBestPath: Boolean,
)

data class SearchGraphSnapshot(
    val runId: String?,
    val title: String,
    val subtitle: String,
    val lifecycleStatus: SearchGraphLifecycleStatus,
    val nodes: List<SearchGraphNodeState>,
    val edges: List<SearchGraphEdgeState>,
    val bestPathIds: List<String>,
    val decisionTrail: List<SearchGraphDecisionStep>,
    val selectedNodeId: String?,
    val newestNodeId: String?,
    val totalNodes: Int,
    val activeNodes: Int,
    val prunedNodes: Int,
    val expandingNodes: Int,
    val successNodeId: String?,
    val bestNodeId: String?,
    val summaryLine: String,
)

class SearchGraphStateStore {
    private val nodes = linkedMapOf<String, SearchGraphNodeState>()
    private val edges = linkedMapOf<String, SearchGraphEdgeState>()
    private var runId: String? = null
    private var title: String = "Search Graph"
    private var subtitle: String = "Waiting for search events."
    private var lifecycleStatus: SearchGraphLifecycleStatus = SearchGraphLifecycleStatus.IDLE
    private var bestPathIds: List<String> = emptyList()
    private var selectedNodeId: String? = null
    private var newestNodeId: String? = null
    private var summaryLine: String = "No search loaded."
    private val decisionTrail = mutableListOf<SearchGraphDecisionStep>()
    private var nextDecisionOrder = 1

    fun clear() {
        nodes.clear()
        edges.clear()
        runId = null
        title = "Search Graph"
        subtitle = "Waiting for search events."
        lifecycleStatus = SearchGraphLifecycleStatus.IDLE
        bestPathIds = emptyList()
        selectedNodeId = null
        newestNodeId = null
        summaryLine = "No search loaded."
        decisionTrail.clear()
        nextDecisionOrder = 1
    }

    fun apply(event: SearchGraphEvent) {
        when (event) {
            is SearchResetEvent -> {
                clear()
                runId = event.runId
                title = "Search Graph"
                subtitle = "Search reset."
                summaryLine = "Search reset."
            }
            is SearchStartedEvent -> {
                clear()
                runId = event.runId
                title = event.title
                subtitle = event.subtitle
                lifecycleStatus = SearchGraphLifecycleStatus.RUNNING
                summaryLine = "Search started."
            }
            is NodeCreatedEvent -> {
                val node = event.node
                nodes[node.id] = SearchGraphNodeState(
                    id = node.id,
                    parentId = node.parentId,
                    depth = node.depth,
                    title = node.title,
                    shortSummary = node.shortSummary,
                    dslSummary = node.dslSummary,
                    patchSummary = node.patchSummary,
                    rationaleSummary = node.rationaleSummary,
                    childIndex = node.childIndex,
                    childCount = node.childCount,
                    status = node.status,
                    createdOrder = node.createdOrder,
                    createdAtLabel = node.createdAtLabel,
                    score = null,
                    rank = null,
                    qValue = null,
                    heuristicScore = null,
                    isBestPath = false,
                    terminalSummary = null,
                    compileStatus = null,
                    testStatus = null,
                    runtimeStatus = null,
                    rawMetadataJson = node.rawMetadataJson,
                )
                newestNodeId = node.id
                if (selectedNodeId == null) {
                    selectedNodeId = node.id
                }
                if (node.status == SearchGraphNodeStatus.ROOT && decisionTrail.isEmpty()) {
                    decisionTrail += SearchGraphDecisionStep(
                        order = nextDecisionOrder++,
                        nodeId = node.id,
                        title = node.title,
                        status = node.status,
                        transitionLabel = "search start",
                        childIndex = node.childIndex,
                        score = null,
                        isBestPath = false,
                    )
                }
                summaryLine = "Created ${node.title}."
            }
            is NodeUpdatedEvent -> {
                val existing = nodes[event.nodeId] ?: return
                nodes[event.nodeId] = existing.copy(
                    title = event.title ?: existing.title,
                    shortSummary = event.shortSummary ?: existing.shortSummary,
                    dslSummary = existing.dslSummary,
                    patchSummary = event.patchSummary ?: existing.patchSummary,
                    rationaleSummary = event.rationaleSummary ?: existing.rationaleSummary,
                    childIndex = event.childIndex ?: existing.childIndex,
                    childCount = event.childCount ?: existing.childCount,
                    rawMetadataJson = event.rawMetadataJson ?: existing.rawMetadataJson,
                )
            }
            is EdgeCreatedEvent -> {
                val edgeId = "${event.parentId}->${event.childId}"
                edges[edgeId] = SearchGraphEdgeState(
                    id = edgeId,
                    parentId = event.parentId,
                    childId = event.childId,
                    actionLabel = event.actionLabel,
                )
            }
            is NodeScoredEvent -> {
                val existing = nodes[event.score.id] ?: return
                nodes[event.score.id] = existing.copy(
                    score = event.score.score ?: existing.score,
                    rank = event.score.rank ?: existing.rank,
                    qValue = event.score.qValue ?: existing.qValue,
                    heuristicScore = event.score.heuristicScore ?: existing.heuristicScore,
                )
                summaryLine = "Scored ${existing.title}."
            }
            is NodePrunedEvent -> {
                val existing = nodes[event.nodeId] ?: return
                nodes[event.nodeId] = existing.copy(
                    status = SearchGraphNodeStatus.PRUNED,
                    terminalSummary = event.reason,
                )
                summaryLine = "Pruned ${existing.title}."
            }
            is NodeStatusChangedEvent -> {
                val existing = nodes[event.nodeId] ?: return
                nodes[event.nodeId] = existing.copy(
                    status = event.status,
                    terminalSummary = event.terminalSummary ?: existing.terminalSummary,
                    compileStatus = event.compileStatus ?: existing.compileStatus,
                    testStatus = event.testStatus ?: existing.testStatus,
                    runtimeStatus = event.runtimeStatus ?: existing.runtimeStatus,
                )
                if (event.status == SearchGraphNodeStatus.EXPANDING) {
                    newestNodeId = event.nodeId
                    decisionTrail += SearchGraphDecisionStep(
                        order = nextDecisionOrder++,
                        nodeId = existing.id,
                        title = existing.title,
                        status = event.status,
                        transitionLabel = event.terminalSummary,
                        childIndex = existing.childIndex,
                        score = existing.score ?: existing.heuristicScore,
                        isBestPath = existing.isBestPath,
                    )
                }
                summaryLine = "${existing.title}: ${event.status.name.lowercase().replace('_', ' ')}."
            }
            is BestPathUpdatedEvent -> {
                bestPathIds = event.nodeIds
                nodes.replaceAll { _, node ->
                    node.copy(isBestPath = node.id in bestPathIds)
                }
                for (index in decisionTrail.indices) {
                    val step = decisionTrail[index]
                    decisionTrail[index] = step.copy(isBestPath = step.nodeId in bestPathIds)
                }
                newestNodeId = event.nodeIds.lastOrNull() ?: newestNodeId
                summaryLine = if (bestPathIds.isEmpty()) "Best path cleared." else "Best path updated."
            }
            is SearchFinishedEvent -> {
                lifecycleStatus = if (event.success) SearchGraphLifecycleStatus.FINISHED else SearchGraphLifecycleStatus.FAILED
                event.terminalNodeId?.let { terminalId ->
                    val existing = nodes[terminalId]
                    if (existing != null) {
                        nodes[terminalId] = existing.copy(
                            status = if (event.success) SearchGraphNodeStatus.SUCCESS else existing.status,
                            terminalSummary = event.summary,
                        )
                    }
                }
                summaryLine = event.summary
            }
        }
    }

    fun selectNode(nodeId: String?) {
        selectedNodeId = nodeId
    }

    fun selectedNode(): SearchGraphNodeState? = selectedNodeId?.let(nodes::get)

    fun loadSnapshot(snapshot: SearchGraphSnapshot) {
        clear()
        runId = snapshot.runId
        title = snapshot.title
        subtitle = snapshot.subtitle
        lifecycleStatus = snapshot.lifecycleStatus
        snapshot.nodes.forEach { node -> nodes[node.id] = node }
        snapshot.edges.forEach { edge -> edges[edge.id] = edge }
        bestPathIds = snapshot.bestPathIds
        selectedNodeId = snapshot.selectedNodeId
        newestNodeId = snapshot.newestNodeId
        summaryLine = snapshot.summaryLine
        decisionTrail += snapshot.decisionTrail
        nextDecisionOrder = (snapshot.decisionTrail.maxOfOrNull { it.order } ?: 0) + 1
    }

    fun snapshot(
        showPruned: Boolean,
        bestPathOnly: Boolean,
    ): SearchGraphSnapshot {
        val effectiveBestPathOnly = bestPathOnly && bestPathIds.isNotEmpty()
        val visibleIds = when {
            effectiveBestPathOnly -> bestPathIds.toSet()
            showPruned -> nodes.keys
            else -> nodes.values.filterNot { it.status == SearchGraphNodeStatus.PRUNED }.map { it.id }.toSet()
        }
        val visibleNodes = nodes.values.filter { it.id in visibleIds }
        val visibleEdges = edges.values.filter { it.parentId in visibleIds && it.childId in visibleIds }
        val successNodeId = nodes.values.firstOrNull { it.status == SearchGraphNodeStatus.SUCCESS }?.id
        return SearchGraphSnapshot(
            runId = runId,
            title = title,
            subtitle = subtitle,
            lifecycleStatus = lifecycleStatus,
            nodes = visibleNodes,
            edges = visibleEdges,
            bestPathIds = bestPathIds.filter { it in visibleIds },
            decisionTrail = decisionTrail.toList(),
            selectedNodeId = selectedNodeId,
            newestNodeId = newestNodeId,
            totalNodes = nodes.size,
            activeNodes = nodes.values.count { it.status in setOf(SearchGraphNodeStatus.ACTIVE, SearchGraphNodeStatus.EXPANDING, SearchGraphNodeStatus.SUCCESS) },
            prunedNodes = nodes.values.count { it.status == SearchGraphNodeStatus.PRUNED },
            expandingNodes = nodes.values.count { it.status == SearchGraphNodeStatus.EXPANDING },
            successNodeId = successNodeId,
            bestNodeId = bestPathIds.lastOrNull(),
            summaryLine = summaryLine,
        )
    }
}
