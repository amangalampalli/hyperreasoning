package com.hyperreasoning.intellij.backend

import com.intellij.openapi.components.Service
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.project.Project
import javax.swing.SwingUtilities

/**
 * Local backend connection/status service for the plugin scaffold.
 *
 * Current responsibilities:
 * - query backend health
 * - submit single-task search runs
 *
 * Future responsibilities:
 * - compare heuristic vs rainbow from the tool window
 * - fetch richer structured tree/patch views
 */
@Service(Service.Level.PROJECT)
class BackendStatusService(private val project: Project) {
    private val settings = ApplicationManager.getApplication().getService(PluginSettingsService::class.java)

    fun currentStatusSummary(): String = "backend: ${settings.state.backendBaseUrl}"

    fun refreshHealth(onDone: (String) -> Unit) {
        ApplicationManager.getApplication().executeOnPooledThread {
            val text = try {
                val client = HyperreasoningApiClient(settings.state.backendBaseUrl)
                val health = client.health()
                "backend: ${health.status}, llm reachable: ${health.llmReachable}"
            } catch (exc: Exception) {
                "backend error: ${exc.message}"
            }
            SwingUtilities.invokeLater { onDone(text) }
        }
    }

    fun runTask(request: TaskRunRequest, onDone: (Result<TaskRunResponse>) -> Unit) {
        ApplicationManager.getApplication().executeOnPooledThread {
            val result = runCatching {
                val client = HyperreasoningApiClient(settings.state.backendBaseUrl)
                client.runTask(request)
            }
            SwingUtilities.invokeLater { onDone(result) }
        }
    }

    fun compareTask(request: CompareStrategiesRequest, onDone: (Result<CompareStrategiesResponse>) -> Unit) {
        ApplicationManager.getApplication().executeOnPooledThread {
            val result = runCatching {
                val client = HyperreasoningApiClient(settings.state.backendBaseUrl)
                client.compareTask(request)
            }
            SwingUtilities.invokeLater { onDone(result) }
        }
    }

    fun renderTaskPayload(request: TaskRunRequest): String {
        val client = HyperreasoningApiClient(settings.state.backendBaseUrl)
        return client.renderTaskPayload(request)
    }

    fun renderComparePayload(request: CompareStrategiesRequest): String {
        val client = HyperreasoningApiClient(settings.state.backendBaseUrl)
        return client.renderComparePayload(request)
    }

    fun runTaskWithProgress(
        request: TaskRunRequest,
        onProgress: (TaskRunJobStatusResponse) -> Unit,
        onDone: (Result<TaskRunResponse>) -> Unit,
    ) {
        ApplicationManager.getApplication().executeOnPooledThread {
            val result: Result<TaskRunResponse> = runCatching {
                val client = HyperreasoningApiClient(settings.state.backendBaseUrl)
                val accepted = client.startTaskRun(request)
                while (true) {
                    val status = client.getTaskRun(accepted.jobId)
                    SwingUtilities.invokeLater { onProgress(status) }
                    when (status.status) {
                        "completed" -> {
                            break
                        }
                        "failed" -> error(status.progress.error ?: "Task job failed")
                    }
                    Thread.sleep(250)
                }
                val finalStatus = client.getTaskRun(accepted.jobId)
                SwingUtilities.invokeLater { onProgress(finalStatus) }
                finalStatus.result ?: error("Completed task job returned no result")
            }
            SwingUtilities.invokeLater { onDone(result) }
        }
    }

    fun compareTaskWithProgress(
        request: CompareStrategiesRequest,
        onProgress: (CompareJobStatusResponse) -> Unit,
        onDone: (Result<CompareStrategiesResponse>) -> Unit,
    ) {
        ApplicationManager.getApplication().executeOnPooledThread {
            val result: Result<CompareStrategiesResponse> = runCatching {
                val client = HyperreasoningApiClient(settings.state.backendBaseUrl)
                val accepted = client.startCompareTask(request)
                while (true) {
                    val status = client.getCompareTask(accepted.jobId)
                    SwingUtilities.invokeLater { onProgress(status) }
                    when (status.status) {
                        "completed" -> {
                            break
                        }
                        "failed" -> error(status.progress.error ?: "Compare job failed")
                    }
                    Thread.sleep(250)
                }
                val finalStatus = client.getCompareTask(accepted.jobId)
                SwingUtilities.invokeLater { onProgress(finalStatus) }
                finalStatus.result ?: error("Completed compare job returned no result")
            }
            SwingUtilities.invokeLater { onDone(result) }
        }
    }

    fun listRuns(context: ClientContext, onDone: (Result<RunHistoryResponse>) -> Unit) {
        ApplicationManager.getApplication().executeOnPooledThread {
            val result = runCatching {
                val client = HyperreasoningApiClient(settings.state.backendBaseUrl)
                client.listRuns(context)
            }
            SwingUtilities.invokeLater { onDone(result) }
        }
    }

    fun loadRun(runId: String, context: ClientContext, onDone: (Result<RunLoadResponse>) -> Unit) {
        ApplicationManager.getApplication().executeOnPooledThread {
            val result = runCatching {
                val client = HyperreasoningApiClient(settings.state.backendBaseUrl)
                client.loadRun(runId, context)
            }
            SwingUtilities.invokeLater { onDone(result) }
        }
    }

    fun syncRuns(context: ClientContext, onDone: (Result<RunSyncResponse>) -> Unit) {
        ApplicationManager.getApplication().executeOnPooledThread {
            val result = runCatching {
                val client = HyperreasoningApiClient(settings.state.backendBaseUrl)
                client.syncRuns(context)
            }
            SwingUtilities.invokeLater { onDone(result) }
        }
    }
}
