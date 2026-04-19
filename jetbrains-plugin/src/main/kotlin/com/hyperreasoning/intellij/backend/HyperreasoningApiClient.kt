package com.hyperreasoning.intellij.backend

import java.net.URI
import java.net.URLEncoder
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.charset.StandardCharsets
import kotlinx.serialization.json.Json

class HyperreasoningApiClient(private val baseUrl: String) {
    private val client = HttpClient.newBuilder()
        .version(HttpClient.Version.HTTP_1_1)
        .build()
    private val json = Json { ignoreUnknownKeys = true }

    private fun ensureSuccess(response: HttpResponse<String>) {
        if (response.statusCode() !in 200..299) {
            val body = response.body().trim()
            throw IllegalStateException(
                "Backend HTTP ${response.statusCode()}: ${if (body.isBlank()) "<empty body>" else body}"
            )
        }
    }

    fun health(): HealthResponse {
        val request = HttpRequest.newBuilder()
            .uri(URI.create("${baseUrl.trimEnd('/')}/health"))
            .GET()
            .build()
        val response = client.send(request, HttpResponse.BodyHandlers.ofString())
        ensureSuccess(response)
        return json.decodeFromString(response.body())
    }

    fun runTask(requestPayload: TaskRunRequest): TaskRunResponse {
        val payload = json.encodeToString(TaskRunRequest.serializer(), requestPayload)
        if (payload.isBlank()) {
            throw IllegalStateException("Encoded task request payload is blank")
        }
        val request = HttpRequest.newBuilder()
            .uri(URI.create("${baseUrl.trimEnd('/')}/api/task/run"))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8))
            .build()
        val response = client.send(request, HttpResponse.BodyHandlers.ofString())
        ensureSuccess(response)
        return json.decodeFromString(response.body())
    }

    fun startTaskRun(requestPayload: TaskRunRequest): AsyncJobAcceptedResponse {
        val payload = json.encodeToString(TaskRunRequest.serializer(), requestPayload)
        if (payload.isBlank()) {
            throw IllegalStateException("Encoded task request payload is blank")
        }
        val request = HttpRequest.newBuilder()
            .uri(URI.create("${baseUrl.trimEnd('/')}/api/task/run_async"))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8))
            .build()
        val response = client.send(request, HttpResponse.BodyHandlers.ofString())
        ensureSuccess(response)
        return json.decodeFromString(response.body())
    }

    fun getTaskRun(jobId: String): TaskRunJobStatusResponse {
        val request = HttpRequest.newBuilder()
            .uri(URI.create("${baseUrl.trimEnd('/')}/api/task/run_async/$jobId"))
            .header("Accept", "application/json")
            .GET()
            .build()
        val response = client.send(request, HttpResponse.BodyHandlers.ofString())
        ensureSuccess(response)
        return json.decodeFromString(response.body())
    }

    fun compareTask(requestPayload: CompareStrategiesRequest): CompareStrategiesResponse {
        val payload = json.encodeToString(CompareStrategiesRequest.serializer(), requestPayload)
        if (payload.isBlank()) {
            throw IllegalStateException("Encoded compare request payload is blank")
        }
        val request = HttpRequest.newBuilder()
            .uri(URI.create("${baseUrl.trimEnd('/')}/api/task/compare"))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8))
            .build()
        val response = client.send(request, HttpResponse.BodyHandlers.ofString())
        ensureSuccess(response)
        return json.decodeFromString(response.body())
    }

    fun startCompareTask(requestPayload: CompareStrategiesRequest): AsyncJobAcceptedResponse {
        val payload = json.encodeToString(CompareStrategiesRequest.serializer(), requestPayload)
        if (payload.isBlank()) {
            throw IllegalStateException("Encoded compare request payload is blank")
        }
        val request = HttpRequest.newBuilder()
            .uri(URI.create("${baseUrl.trimEnd('/')}/api/task/compare_async"))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8))
            .build()
        val response = client.send(request, HttpResponse.BodyHandlers.ofString())
        ensureSuccess(response)
        return json.decodeFromString(response.body())
    }

    fun getCompareTask(jobId: String): CompareJobStatusResponse {
        val request = HttpRequest.newBuilder()
            .uri(URI.create("${baseUrl.trimEnd('/')}/api/task/compare_async/$jobId"))
            .header("Accept", "application/json")
            .GET()
            .build()
        val response = client.send(request, HttpResponse.BodyHandlers.ofString())
        ensureSuccess(response)
        return json.decodeFromString(response.body())
    }

    fun listRuns(context: ClientContext, limit: Int = 50, offset: Int = 0, query: String? = null): RunHistoryResponse {
        val params = linkedMapOf<String, String>()
        context.projectId?.let { params["project_id"] = it }
        context.projectRoot?.let { params["project_root"] = it }
        context.projectName?.let { params["project_name"] = it }
        context.taskRoot?.let { params["task_root"] = it }
        context.activeFile?.let { params["active_file"] = it }
        params["limit"] = limit.toString()
        params["offset"] = offset.toString()
        query?.takeIf { it.isNotBlank() }?.let { params["query"] = it }
        val queryString = params.entries.joinToString("&") { (key, value) ->
            "${encode(key)}=${encode(value)}"
        }
        val request = HttpRequest.newBuilder()
            .uri(URI.create("${baseUrl.trimEnd('/')}/api/runs?$queryString"))
            .header("Accept", "application/json")
            .GET()
            .build()
        val response = client.send(request, HttpResponse.BodyHandlers.ofString())
        ensureSuccess(response)
        return json.decodeFromString(response.body())
    }

    fun loadRun(runId: String, context: ClientContext): RunLoadResponse {
        val params = linkedMapOf<String, String>()
        context.projectId?.let { params["project_id"] = it }
        context.projectRoot?.let { params["project_root"] = it }
        context.projectName?.let { params["project_name"] = it }
        context.taskRoot?.let { params["task_root"] = it }
        context.activeFile?.let { params["active_file"] = it }
        val queryString = params.entries.joinToString("&") { (key, value) ->
            "${encode(key)}=${encode(value)}"
        }
        val suffix = if (queryString.isBlank()) "" else "?$queryString"
        val request = HttpRequest.newBuilder()
            .uri(URI.create("${baseUrl.trimEnd('/')}/api/runs/${encode(runId)}$suffix"))
            .header("Accept", "application/json")
            .GET()
            .build()
        val response = client.send(request, HttpResponse.BodyHandlers.ofString())
        ensureSuccess(response)
        return json.decodeFromString(response.body())
    }

    fun syncRuns(context: ClientContext, limit: Int = 100): RunSyncResponse {
        val payload = json.encodeToString(RunSyncRequest.serializer(), RunSyncRequest(context, limit))
        val request = HttpRequest.newBuilder()
            .uri(URI.create("${baseUrl.trimEnd('/')}/api/runs/sync"))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8))
            .build()
        val response = client.send(request, HttpResponse.BodyHandlers.ofString())
        ensureSuccess(response)
        return json.decodeFromString(response.body())
    }

    fun renderTaskPayload(requestPayload: TaskRunRequest): String {
        return json.encodeToString(TaskRunRequest.serializer(), requestPayload)
    }

    fun renderComparePayload(requestPayload: CompareStrategiesRequest): String {
        return json.encodeToString(CompareStrategiesRequest.serializer(), requestPayload)
    }

    private fun encode(value: String): String {
        return URLEncoder.encode(value, StandardCharsets.UTF_8)
    }
}
