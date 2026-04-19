package com.hyperreasoning.intellij.toolwindow

import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

object SearchGraphBackendAdapter {
    fun decode(events: List<Map<String, JsonElement>>): List<SearchGraphEvent> {
        return events.mapNotNull(::decodeEvent)
    }

    private fun decodeEvent(raw: Map<String, JsonElement>): SearchGraphEvent? {
        return when (raw.string("type")) {
            "search_started" -> SearchStartedEvent(
                runId = raw.string("run_id") ?: return null,
                title = raw.string("title") ?: "Search Graph",
                subtitle = raw.string("subtitle") ?: "",
            )
            "search_reset" -> SearchResetEvent(
                runId = raw.string("run_id") ?: return null,
            )
            "node_created" -> {
                val node = raw.jsonObject("node") ?: return null
                NodeCreatedEvent(
                    runId = raw.string("run_id") ?: return null,
                    node = SearchGraphNodePayload(
                        id = node.string("id") ?: return null,
                        parentId = node.string("parentId"),
                        depth = node.int("depth") ?: 0,
                        title = node.string("title") ?: "Node",
                        shortSummary = node.string("shortSummary").orEmpty(),
                        dslSummary = node.string("dslSummary").orEmpty(),
                        patchSummary = node.string("patchSummary").orEmpty(),
                        rationaleSummary = node.string("rationaleSummary").orEmpty(),
                        childIndex = node.int("childIndex"),
                        childCount = node.int("childCount"),
                        status = decodeStatus(node.string("status")),
                        createdOrder = node.int("createdOrder") ?: 0,
                        createdAtLabel = node.string("createdAtLabel"),
                        rawMetadataJson = node.string("rawMetadataJson"),
                    ),
                )
            }
            "node_updated" -> NodeUpdatedEvent(
                runId = raw.string("run_id") ?: return null,
                nodeId = raw.string("node_id") ?: return null,
                title = raw.string("title"),
                shortSummary = raw.string("shortSummary"),
                patchSummary = raw.string("patchSummary"),
                rationaleSummary = raw.string("rationaleSummary"),
                childIndex = raw.int("childIndex"),
                childCount = raw.int("childCount"),
                rawMetadataJson = raw.string("rawMetadataJson"),
            )
            "edge_created" -> EdgeCreatedEvent(
                runId = raw.string("run_id") ?: return null,
                parentId = raw.string("parent_id") ?: return null,
                childId = raw.string("child_id") ?: return null,
                actionLabel = raw.string("action_label"),
            )
            "node_scored" -> {
                val score = raw.jsonObject("score") ?: return null
                NodeScoredEvent(
                    runId = raw.string("run_id") ?: return null,
                    score = SearchGraphScorePayload(
                        id = score.string("id") ?: return null,
                        score = score.double("score"),
                        rank = score.int("rank"),
                        qValue = score.double("qValue"),
                        heuristicScore = score.double("heuristicScore"),
                    ),
                )
            }
            "node_pruned" -> NodePrunedEvent(
                runId = raw.string("run_id") ?: return null,
                nodeId = raw.string("node_id") ?: return null,
                reason = raw.string("reason") ?: "Pruned",
            )
            "node_status_changed" -> NodeStatusChangedEvent(
                runId = raw.string("run_id") ?: return null,
                nodeId = raw.string("node_id") ?: return null,
                status = decodeStatus(raw.string("status")),
                terminalSummary = raw.string("terminal_summary"),
                compileStatus = raw.string("compile_status"),
                testStatus = raw.string("test_status"),
                runtimeStatus = raw.string("runtime_status"),
            )
            "best_path_updated" -> BestPathUpdatedEvent(
                runId = raw.string("run_id") ?: return null,
                nodeIds = raw.jsonArrayStrings("node_ids"),
            )
            "search_finished" -> SearchFinishedEvent(
                runId = raw.string("run_id") ?: return null,
                terminalNodeId = raw.string("terminal_node_id"),
                success = raw.bool("success") ?: false,
                summary = raw.string("summary") ?: "Search finished.",
            )
            else -> null
        }
    }

    private fun decodeStatus(raw: String?): SearchGraphNodeStatus {
        return when (raw?.uppercase()) {
            "ROOT" -> SearchGraphNodeStatus.ROOT
            "ACTIVE" -> SearchGraphNodeStatus.ACTIVE
            "EXPANDING" -> SearchGraphNodeStatus.EXPANDING
            "PRUNED" -> SearchGraphNodeStatus.PRUNED
            "SUCCESS" -> SearchGraphNodeStatus.SUCCESS
            "FAILED_COMPILE" -> SearchGraphNodeStatus.FAILED_COMPILE
            "FAILED_TEST" -> SearchGraphNodeStatus.FAILED_TEST
            "FAILED_RUNTIME" -> SearchGraphNodeStatus.FAILED_RUNTIME
            else -> SearchGraphNodeStatus.IDLE
        }
    }
}

private fun Map<String, JsonElement>.string(key: String): String? = (this[key] as? JsonPrimitive)?.contentOrNull()
private fun Map<String, JsonElement>.int(key: String): Int? = (this[key] as? JsonPrimitive)?.content?.toIntOrNull()
private fun Map<String, JsonElement>.double(key: String): Double? = (this[key] as? JsonPrimitive)?.content?.toDoubleOrNull()
private fun Map<String, JsonElement>.bool(key: String): Boolean? = when ((this[key] as? JsonPrimitive)?.content?.lowercase()) {
    "true" -> true
    "false" -> false
    else -> null
}
private fun Map<String, JsonElement>.jsonObject(key: String): Map<String, JsonElement>? = (this[key] as? JsonObject)?.toMap()
private fun Map<String, JsonElement>.jsonArrayStrings(key: String): List<String> {
    val value = this[key] ?: return emptyList()
    return if (value is kotlinx.serialization.json.JsonArray) {
        value.mapNotNull { (it as? JsonPrimitive)?.contentOrNull() }
    } else {
        emptyList()
    }
}

private fun JsonPrimitive.contentOrNull(): String? {
    val value = content
    return value.takeUnless { !isString && value == "null" }
}
