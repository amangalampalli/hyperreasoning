package com.hyperreasoning.intellij.toolwindow

import com.hyperreasoning.intellij.backend.TaskRunResponse
import com.intellij.openapi.Disposable
import javax.swing.Timer
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

data class TimedSearchGraphEvent(
    val delayMs: Int,
    val event: SearchGraphEvent,
)

interface SearchGraphEventSubscription : Disposable {
    val supportsPlayback: Boolean
    fun play()
    fun pause()
    fun reset()
    fun step()
}

interface SearchGraphEventSource {
    fun subscribeToSearchGraphEvents(
        runId: String,
        listener: (SearchGraphEvent) -> Unit,
    ): SearchGraphEventSubscription
}

class PlaceholderLiveSearchGraphEventSource(
    private val connect: (runId: String, listener: (SearchGraphEvent) -> Unit) -> Disposable,
) : SearchGraphEventSource {
    override fun subscribeToSearchGraphEvents(
        runId: String,
        listener: (SearchGraphEvent) -> Unit,
    ): SearchGraphEventSubscription {
        val disposable = connect(runId, listener)
        return object : SearchGraphEventSubscription {
            override val supportsPlayback: Boolean = false
            override fun play() = Unit
            override fun pause() = Unit
            override fun reset() = Unit
            override fun step() = Unit
            override fun dispose() {
                disposable.dispose()
            }
        }
    }
}

private class ReplaySearchGraphEventSubscription(
    private val runId: String,
    private val events: List<TimedSearchGraphEvent>,
    private val listener: (SearchGraphEvent) -> Unit,
) : SearchGraphEventSubscription {
    private var cursor = 0
    private var timer: Timer? = null

    override val supportsPlayback: Boolean = true

    override fun play() {
        if (cursor >= events.size) {
            return
        }
        scheduleNext()
    }

    override fun pause() {
        timer?.stop()
        timer = null
    }

    override fun reset() {
        pause()
        cursor = 0
        listener(SearchResetEvent(runId))
    }

    override fun step() {
        pause()
        if (cursor < events.size) {
            val next = events[cursor]
            cursor += 1
            listener(next.event)
        }
    }

    override fun dispose() {
        pause()
    }

    private fun scheduleNext() {
        if (cursor >= events.size) {
            pause()
            return
        }
        val next = events[cursor]
        timer?.stop()
        timer = Timer(next.delayMs.coerceAtLeast(1)) {
            timer?.stop()
            timer = null
            listener(next.event)
            cursor += 1
            scheduleNext()
        }.apply {
            isRepeats = false
            start()
        }
    }
}

private class ReplaySearchGraphEventSource(
    private val eventFactory: (String) -> List<TimedSearchGraphEvent>,
) : SearchGraphEventSource {
    override fun subscribeToSearchGraphEvents(
        runId: String,
        listener: (SearchGraphEvent) -> Unit,
    ): SearchGraphEventSubscription {
        return ReplaySearchGraphEventSubscription(runId, eventFactory(runId), listener)
    }
}

object SearchGraphEventSources {
    fun mockDemoSource(): SearchGraphEventSource {
        return ReplaySearchGraphEventSource { runId -> buildMockDemoEvents(runId) }
    }

    fun recordedRunEvents(
        runId: String,
        label: String,
        response: TaskRunResponse,
    ): List<SearchGraphEvent> {
        return buildRecordedEvents(runId, label, response).map { it.event }
    }

    fun recordedRunSource(
        label: String,
        response: TaskRunResponse,
    ): SearchGraphEventSource {
        return ReplaySearchGraphEventSource { runId -> buildRecordedEvents(runId, label, response) }
    }

    private fun buildMockDemoEvents(runId: String): List<TimedSearchGraphEvent> {
        val events = mutableListOf<TimedSearchGraphEvent>()
        fun emit(delayMs: Int, event: SearchGraphEvent) {
            events += TimedSearchGraphEvent(delayMs, event)
        }

        emit(60, SearchStartedEvent(runId, "Search Graph Demo", "Mock live search replay for hackathon demos."))
        emit(
            90,
            NodeCreatedEvent(
                runId,
                SearchGraphNodePayload(
                    id = "root",
                    title = "ROOT",
                    status = SearchGraphNodeStatus.ROOT,
                    createdOrder = 0,
                    shortSummary = "Initial problem state",
                    dslSummary = "root_state",
                    patchSummary = "No patch applied yet",
                ),
            ),
        )

        val candidates = listOf(
            Triple("n1", "Minimal patch", 0.46),
            Triple("n2", "Refactor guard", 0.72),
            Triple("n3", "Algorithm swap", 0.38),
        )
        candidates.forEachIndexed { index, (id, title, score) ->
            emit(
                120,
                NodeCreatedEvent(
                    runId,
                    SearchGraphNodePayload(
                        id = id,
                        parentId = "root",
                        depth = 1,
                        title = title,
                        shortSummary = "Root child ${index + 1}",
                        dslSummary = "strategy=${title.lowercase().replace(' ', '_')}",
                        patchSummary = "Candidate branch ${index + 1}",
                        rationaleSummary = "Generated from initial search expansion",
                        childIndex = index + 1,
                        childCount = candidates.size,
                        status = SearchGraphNodeStatus.ACTIVE,
                        createdOrder = index + 1,
                    ),
                ),
            )
            emit(20, EdgeCreatedEvent(runId, "root", id, "SELECT_CHILD_$index"))
            emit(30, NodeScoredEvent(runId, SearchGraphScorePayload(id = id, score = score, heuristicScore = score, rank = index + 1)))
        }

        emit(140, NodeStatusChangedEvent(runId, "n2", SearchGraphNodeStatus.EXPANDING, terminalSummary = "Current best candidate is expanding."))
        emit(20, BestPathUpdatedEvent(runId, listOf("root", "n2")))

        val branchTwoChildren = listOf(
            Triple("n4", "Tighten null check", 0.66),
            Triple("n5", "Rebuild parse flow", 0.81),
            Triple("n6", "Patch timeout path", 0.41),
        )
        branchTwoChildren.forEachIndexed { index, (id, title, score) ->
            emit(
                110,
                NodeCreatedEvent(
                    runId,
                    SearchGraphNodePayload(
                        id = id,
                        parentId = "n2",
                        depth = 2,
                        title = title,
                        shortSummary = "Child ${index + 1} under Refactor guard",
                        dslSummary = "strategy=${title.lowercase().replace(' ', '_')}",
                        patchSummary = "Refinement of the current best branch",
                        rationaleSummary = "Expanded after the ranker promoted branch n2",
                        childIndex = index + 1,
                        childCount = branchTwoChildren.size,
                        status = SearchGraphNodeStatus.ACTIVE,
                        createdOrder = 10 + index,
                    ),
                ),
            )
            emit(20, EdgeCreatedEvent(runId, "n2", id, "SELECT_CHILD_$index"))
            emit(30, NodeScoredEvent(runId, SearchGraphScorePayload(id = id, score = score, heuristicScore = score, rank = index + 1)))
        }

        emit(100, NodePrunedEvent(runId, "n1", "Low score after first ranking pass"))
        emit(80, NodeStatusChangedEvent(runId, "n4", SearchGraphNodeStatus.FAILED_TEST, terminalSummary = "Visible tests failed", testStatus = "visible test failed"))
        emit(80, NodeStatusChangedEvent(runId, "n6", SearchGraphNodeStatus.PRUNED, terminalSummary = "Dominated by stronger sibling"))
        emit(80, NodeStatusChangedEvent(runId, "n5", SearchGraphNodeStatus.EXPANDING, terminalSummary = "Now expanding the most promising child"))
        emit(20, BestPathUpdatedEvent(runId, listOf("root", "n2", "n5")))

        emit(
            120,
            NodeCreatedEvent(
                runId,
                SearchGraphNodePayload(
                    id = "n7",
                    parentId = "n5",
                    depth = 3,
                    title = "Validated patch",
                    shortSummary = "Compiled and passed visible tests",
                    dslSummary = "strategy=validated_patch\nchecks=visible_tests",
                    patchSummary = "Winning patch candidate",
                    rationaleSummary = "Best branch continued to a successful verification",
                    childIndex = 1,
                    childCount = 1,
                    status = SearchGraphNodeStatus.SUCCESS,
                    createdOrder = 20,
                ),
            ),
        )
        emit(20, EdgeCreatedEvent(runId, "n5", "n7", "COMPILE_TO_CODE"))
        emit(30, NodeScoredEvent(runId, SearchGraphScorePayload(id = "n7", score = 0.97, heuristicScore = 0.97, rank = 1)))
        emit(30, BestPathUpdatedEvent(runId, listOf("root", "n2", "n5", "n7")))
        emit(40, SearchFinishedEvent(runId, terminalNodeId = "n7", success = true, summary = "Demo search finished with a successful solution node."))

        return events
    }

    private fun buildRecordedEvents(
        runId: String,
        label: String,
        response: TaskRunResponse,
    ): List<TimedSearchGraphEvent> {
        val events = mutableListOf<TimedSearchGraphEvent>()
        fun emit(delayMs: Int, event: SearchGraphEvent) {
            events += TimedSearchGraphEvent(delayMs, event)
        }

        val planBank = response.planBank
        val entries = planBank.jsonObject("entries").orEmpty()
        val rootIds = planBank.jsonArray("root_bank_ids")?.mapNotNull { (it as? JsonPrimitive)?.content }.orEmpty()
        val siblingInfo = buildSiblingInfo(rootIds, entries)
        val resultByBank = buildResultByBank(response)
        val created = mutableSetOf("root")
        val title = "${label} Search"
        emit(60, SearchStartedEvent(runId, title, "Recorded replay derived from the current task run response."))
        emit(
            70,
            NodeCreatedEvent(
                runId,
                SearchGraphNodePayload(
                    id = "root",
                    title = "ROOT",
                    status = SearchGraphNodeStatus.ROOT,
                    shortSummary = "Initial search state",
                    dslSummary = "root_state",
                    patchSummary = "Current task root",
                    createdOrder = 0,
                ),
            ),
        )
        fun emitBankNode(bankId: String) {
            if (!created.add(bankId)) {
                return
            }
            val entry = entries[bankId] as? JsonObject ?: return
            val parentBankId = entry.string("parent_bank_id")
            val strategyTitle = strategyTitle(entry.jsonObject("plan"), bankId)
            val slot = siblingInfo[bankId]
            emit(
                36,
                NodeCreatedEvent(
                    runId,
                    SearchGraphNodePayload(
                        id = bankId,
                        parentId = parentBankId ?: "root",
                        depth = (entry.int("depth") ?: 0) + 1,
                        title = strategyTitle,
                        shortSummary = compactSummary(entry.jsonObject("plan")),
                        dslSummary = dslSummary(entry.jsonObject("plan")),
                        patchSummary = patchSummary(entry.jsonObject("plan")),
                        rationaleSummary = entry.jsonObject("plan")?.string("notes").orEmpty(),
                        childIndex = slot?.first,
                        childCount = slot?.second,
                        status = SearchGraphNodeStatus.IDLE,
                        createdOrder = created.size,
                        rawMetadataJson = entry.toString(),
                    ),
                ),
            )
            emit(12, EdgeCreatedEvent(runId, parentBankId ?: "root", bankId, actionLabel = null))
            entry.double("heuristic_score")?.let { score ->
                emit(
                    12,
                    NodeScoredEvent(
                        runId,
                        SearchGraphScorePayload(
                            id = bankId,
                            score = score,
                            heuristicScore = score,
                        ),
                    ),
                )
            }
        }

        response.transitions.firstOrNull()
            ?.jsonObject("state")
            ?.childSlotBankIds()
            ?.forEach(::emitBankNode)

        val traceProgress = mutableListOf("root")
        val touchedBankIds = linkedSetOf<String>()
        response.transitions.forEach { transition ->
            val stateBefore = transition.jsonObject("state")
            val nextState = transition.jsonObject("next_state")
            stateBefore?.childSlotBankIds()?.forEach(::emitBankNode)
            val action = transition.string("action")
            val currentBankId = nextState?.string("current_bank_id")
            val currentBefore = stateBefore?.string("current_bank_id")
            val info = transition.jsonObject("info")

            when {
                action?.startsWith("SELECT_CHILD_") == true -> {
                    currentBankId?.let { bankId ->
                        emitBankNode(bankId)
                        touchedBankIds += bankId
                        emit(
                            48,
                            NodeStatusChangedEvent(
                                runId,
                                bankId,
                                SearchGraphNodeStatus.EXPANDING,
                                terminalSummary = describeAction(action),
                            ),
                        )
                        traceProgress += bankId
                        emit(16, BestPathUpdatedEvent(runId, traceProgress.toList()))
                    }
                }
                action in setOf("REQUEST_MORE_CANDIDATES", "REFINE_CURRENT_PLAN") -> {
                    nextState?.childSlotBankIds()?.forEach(::emitBankNode)
                    currentBefore?.let { bankId ->
                        touchedBankIds += bankId
                        emit(
                            36,
                            NodeStatusChangedEvent(
                                runId,
                                bankId,
                                SearchGraphNodeStatus.EXPANDING,
                                terminalSummary = action?.let(::describeAction),
                            ),
                        )
                    }
                }
                action == "BACKTRACK" -> {
                    currentBankId?.let { bankId ->
                        touchedBankIds += bankId
                        emit(
                            36,
                            NodeStatusChangedEvent(
                                runId,
                                bankId,
                                SearchGraphNodeStatus.EXPANDING,
                                terminalSummary = describeAction(action),
                            ),
                        )
                        if (traceProgress.lastOrNull() != bankId) {
                            traceProgress += bankId
                            emit(16, BestPathUpdatedEvent(runId, traceProgress.toList()))
                        }
                    }
                }
            }

            val resultBankId = currentBefore ?: currentBankId
            if (resultBankId != null && info != null && listOf("compile_success", "visible_test_passed", "hidden_test_passed").any { info[it] != null }) {
                touchedBankIds += resultBankId
                emitBankNode(resultBankId)
                emit(
                    18,
                    NodeStatusChangedEvent(
                        runId,
                        resultBankId,
                        nodeStatusForResult(
                            Triple(
                                info.bool("compile_success"),
                                info.bool("visible_test_passed"),
                                info.bool("hidden_test_passed"),
                            ),
                            default = SearchGraphNodeStatus.ACTIVE,
                        ),
                        terminalSummary = terminalSummary(
                            Triple(
                                info.bool("compile_success"),
                                info.bool("visible_test_passed"),
                                info.bool("hidden_test_passed"),
                            ),
                        ),
                        compileStatus = compileStatus(
                            Triple(
                                info.bool("compile_success"),
                                info.bool("visible_test_passed"),
                                info.bool("hidden_test_passed"),
                            ),
                        ),
                        testStatus = testStatus(
                            Triple(
                                info.bool("compile_success"),
                                info.bool("visible_test_passed"),
                                info.bool("hidden_test_passed"),
                            ),
                        ),
                    ),
                )
            }
        }

        created
            .filter { it != "root" && it !in touchedBankIds && it !in traceProgress }
            .forEach { bankId ->
                emit(12, NodePrunedEvent(runId, bankId, "Visible during search but never promoted"))
            }

        touchedBankIds.forEach { bankId ->
            val status = nodeStatusForResult(resultByBank[bankId], default = SearchGraphNodeStatus.ACTIVE)
            emit(
                12,
                NodeStatusChangedEvent(
                    runId,
                    bankId,
                    status,
                    terminalSummary = terminalSummary(resultByBank[bankId]),
                    compileStatus = compileStatus(resultByBank[bankId]),
                    testStatus = testStatus(resultByBank[bankId]),
                ),
            )
        }

        val terminalBankId = response.strategy.bestBankId ?: traceProgress.lastOrNull()
        if (terminalBankId != null && response.strategy.visibleTestPassed == true) {
            emit(
                24,
                NodeStatusChangedEvent(
                    runId,
                    terminalBankId,
                    SearchGraphNodeStatus.SUCCESS,
                    terminalSummary = "Visible tests passed for the current best candidate",
                    compileStatus = "compiled",
                    testStatus = "visible tests passed",
                ),
            )
        }
        emit(24, BestPathUpdatedEvent(runId, traceProgress.toList()))
        emit(
            30,
            SearchFinishedEvent(
                runId,
                terminalNodeId = terminalBankId,
                success = response.strategy.visibleTestPassed == true || response.strategy.compileSuccess == true,
                summary = if (response.strategy.visibleTestPassed == true) {
                    "${label} replay finished with a successful verified node."
                } else {
                    "${label} replay finished. Best node: ${terminalBankId ?: "-"}."
                },
            ),
        )

        return events
    }

    private fun buildSiblingInfo(
        rootIds: List<String>,
        entries: Map<String, JsonElement>,
    ): Map<String, Pair<Int, Int>> {
        val info = mutableMapOf<String, Pair<Int, Int>>()
        rootIds.forEachIndexed { index, bankId ->
            info[bankId] = (index + 1) to rootIds.size
        }
        entries.values.forEach { rawEntry ->
            val entry = rawEntry as? JsonObject ?: return@forEach
            val children = entry.jsonArray("child_bank_ids")?.mapNotNull { (it as? JsonPrimitive)?.content }.orEmpty()
            children.forEachIndexed { index, childId ->
                info[childId] = (index + 1) to children.size
            }
        }
        return info
    }

    private fun buildResultByBank(
        response: TaskRunResponse,
    ): Map<String, Triple<Boolean?, Boolean?, Boolean?>> {
        val resultByBank = mutableMapOf<String, Triple<Boolean?, Boolean?, Boolean?>>()
        response.nodes.forEach { raw ->
            val bankId = raw.bankId() ?: return@forEach
            resultByBank[bankId] = Triple(
                raw.bool("compile_success"),
                raw.bool("visible_test_passed"),
                raw.bool("hidden_test_passed"),
            )
        }
        return resultByBank
    }

    private fun nodeStatusForResult(
        result: Triple<Boolean?, Boolean?, Boolean?>?,
        default: SearchGraphNodeStatus,
    ): SearchGraphNodeStatus {
        return when {
            result == null -> default
            result.second == true || result.third == true -> SearchGraphNodeStatus.SUCCESS
            result.second == false -> SearchGraphNodeStatus.FAILED_TEST
            result.first == false -> SearchGraphNodeStatus.FAILED_COMPILE
            result.first == true -> SearchGraphNodeStatus.ACTIVE
            else -> default
        }
    }

    private fun terminalSummary(result: Triple<Boolean?, Boolean?, Boolean?>?): String? {
        return when {
            result == null -> null
            result.second == true || result.third == true -> "Tests passed"
            result.second == false -> "Visible tests failed"
            result.first == false -> "Compile failed"
            result.first == true -> "Compiled"
            else -> null
        }
    }

    private fun compileStatus(result: Triple<Boolean?, Boolean?, Boolean?>?): String? {
        return when (result?.first) {
            true -> "compiled"
            false -> "compile failed"
            null -> null
        }
    }

    private fun testStatus(result: Triple<Boolean?, Boolean?, Boolean?>?): String? {
        return when {
            result?.third == true -> "hidden tests passed"
            result?.second == true -> "visible tests passed"
            result?.second == false -> "visible tests failed"
            else -> null
        }
    }

    private fun compactSummary(plan: JsonObject?): String {
        if (plan == null) return ""
        val bugs = plan.jsonArray("suspected_bug_types")
            ?.mapNotNull { (it as? JsonPrimitive)?.content }
            .takeUnless { it.isNullOrEmpty() }
            ?.joinToString(limit = 2)
        return buildString {
            append(plan.string("strategy") ?: "candidate")
            bugs?.let { append(" • ").append(it) }
        }
    }

    private fun patchSummary(plan: JsonObject?): String {
        if (plan == null) return ""
        val targets = plan.jsonArray("target_files")
            ?.mapNotNull { (it as? JsonPrimitive)?.content }
            .orEmpty()
            .joinToString(limit = 3)
        return if (targets.isBlank()) "Patch candidate" else "Targets $targets"
    }

    private fun dslSummary(plan: JsonObject?): String {
        if (plan == null) return ""
        val lines = mutableListOf<String>()
        plan.string("strategy")?.let { lines += "strategy=$it" }
        plan.jsonArray("target_files")
            ?.mapNotNull { (it as? JsonPrimitive)?.content }
            ?.takeIf { it.isNotEmpty() }
            ?.let { lines += "files=${it.joinToString()}" }
        plan.jsonArray("suspected_bug_types")
            ?.mapNotNull { (it as? JsonPrimitive)?.content }
            ?.takeIf { it.isNotEmpty() }
            ?.let { lines += "bugs=${it.joinToString()}" }
        plan.jsonArray("validation_checks")
            ?.mapNotNull { (it as? JsonPrimitive)?.content }
            ?.takeIf { it.isNotEmpty() }
            ?.let { lines += "checks=${it.joinToString()}" }
        plan.string("notes")?.takeIf { it.isNotBlank() }?.let { lines += "notes=$it" }
        return lines.joinToString("\n")
    }

    private fun strategyTitle(plan: JsonObject?, bankId: String): String {
        val strategy = plan?.string("strategy")?.replace('_', ' ')?.replaceFirstChar { it.uppercase() }
        return strategy ?: bankId.takeLast(4)
    }

    private fun describeAction(action: String): String {
        return when {
            action.startsWith("SELECT_CHILD_") -> {
                val slot = action.substringAfterLast('_').toIntOrNull()?.plus(1) ?: "?"
                "selected child $slot"
            }
            else -> action.replace('_', ' ').lowercase()
        }
    }
}

private fun Map<String, JsonElement>.bankId(): String? = string("bank_id") ?: jsonObject("state")?.string("current_bank_id")
private fun Map<String, JsonElement>.string(key: String): String? = (this[key] as? JsonPrimitive)?.content
private fun Map<String, JsonElement>.bool(key: String): Boolean? = when ((this[key] as? JsonPrimitive)?.content?.lowercase()) {
    "true" -> true
    "false" -> false
    else -> null
}
private fun Map<String, JsonElement>.jsonObject(key: String): JsonObject? = this[key] as? JsonObject
private fun Map<String, JsonElement>.jsonArray(key: String): JsonArray? = this[key] as? JsonArray
private fun JsonObject.string(key: String): String? = (this[key] as? JsonPrimitive)?.content
private fun JsonObject.bool(key: String): Boolean? = when ((this[key] as? JsonPrimitive)?.content?.lowercase()) {
    "true" -> true
    "false" -> false
    else -> null
}
private fun JsonObject.double(key: String): Double? = (this[key] as? JsonPrimitive)?.content?.toDoubleOrNull()
private fun JsonObject.int(key: String): Int? = (this[key] as? JsonPrimitive)?.content?.toIntOrNull()
private fun JsonObject.jsonArray(key: String): JsonArray? = this[key] as? JsonArray
private fun JsonObject.childSlotBankIds(): List<String> = jsonArray("child_slots")
    ?.mapNotNull { (it as? JsonObject)?.string("bank_id") }
    .orEmpty()
