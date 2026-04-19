package com.hyperreasoning.intellij.backend

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class HealthResponse(
    val status: String,
    @SerialName("llm_base_url") val llmBaseUrl: String,
    @SerialName("llm_reachable") val llmReachable: Boolean,
)

@Serializable
data class ClientContext(
    @SerialName("project_id") val projectId: String? = null,
    @SerialName("project_name") val projectName: String? = null,
    @SerialName("project_root") val projectRoot: String? = null,
    @SerialName("task_root") val taskRoot: String? = null,
    @SerialName("active_file") val activeFile: String? = null,
)

@Serializable
data class TaskRunRequest(
    @SerialName("client_context") val clientContext: ClientContext? = null,
    val prompt: String,
    val files: Map<String, String>,
    @SerialName("target_files") val targetFiles: List<String>,
    @SerialName("visible_test_file") val visibleTestFile: String? = null,
    @SerialName("hidden_test_file") val hiddenTestFile: String? = null,
    val language: String = "python",
    val family: String = "custom_single_file",
    val policy: String = "rainbow",
    @SerialName("proposal_source") val proposalSource: String = "heuristic",
    @SerialName("max_steps") val maxSteps: Int = 8,
    @SerialName("max_verified_plans_per_task") val maxVerifiedPlansPerTask: Int = 1,
    @SerialName("allow_full_file_fallback") val allowFullFileFallback: Boolean = true,
    @SerialName("run_tests") val runTests: Boolean = true,
    @SerialName("run_hidden_tests") val runHiddenTests: Boolean = false,
    @SerialName("checkpoint_path") val checkpointPath: String? = null,
    val seed: Int = 123,
)

@Serializable
data class CompareStrategiesRequest(
    @SerialName("client_context") val clientContext: ClientContext? = null,
    val prompt: String,
    val files: Map<String, String>,
    @SerialName("target_files") val targetFiles: List<String>,
    @SerialName("visible_test_file") val visibleTestFile: String? = null,
    @SerialName("hidden_test_file") val hiddenTestFile: String? = null,
    val language: String = "python",
    val family: String = "custom_single_file",
    @SerialName("proposal_source") val proposalSource: String = "heuristic",
    @SerialName("max_steps") val maxSteps: Int = 8,
    @SerialName("max_verified_plans_per_task") val maxVerifiedPlansPerTask: Int = 1,
    @SerialName("allow_full_file_fallback") val allowFullFileFallback: Boolean = true,
    @SerialName("run_tests") val runTests: Boolean = true,
    @SerialName("run_hidden_tests") val runHiddenTests: Boolean = false,
    @SerialName("checkpoint_path") val checkpointPath: String? = null,
    val seed: Int = 123,
    val policies: List<String> = listOf("heuristic", "rainbow", "oneshot"),
)

@Serializable
data class StrategySummary(
    val policy: String,
    @SerialName("task_id") val taskId: String,
    val family: String,
    @SerialName("total_reward") val totalReward: Double,
    val steps: Int,
    @SerialName("compile_successes") val compileSuccesses: Int,
    @SerialName("visible_passes") val visiblePasses: Int,
    @SerialName("hidden_passes") val hiddenPasses: Int = 0,
    @SerialName("visible_tests_passed") val visibleTestsPassed: Int = 0,
    @SerialName("visible_tests_total") val visibleTestsTotal: Int = 0,
    @SerialName("hidden_tests_passed") val hiddenTestsPassed: Int = 0,
    @SerialName("hidden_tests_total") val hiddenTestsTotal: Int = 0,
    @SerialName("tests_passed") val testsPassed: Int = 0,
    @SerialName("tests_total") val testsTotal: Int = 0,
    @SerialName("fraction_tests_passed") val fractionTestsPassed: Double = 0.0,
    @SerialName("best_bank_id") val bestBankId: String? = null,
    @SerialName("compile_success") val compileSuccess: Boolean? = null,
    @SerialName("visible_test_passed") val visibleTestPassed: Boolean? = null,
    @SerialName("hidden_test_passed") val hiddenTestPassed: Boolean? = null,
    @SerialName("compile_error") val compileError: String? = null,
    @SerialName("hidden_test_stdout") val hiddenTestStdout: String? = null,
    @SerialName("hidden_test_stderr") val hiddenTestStderr: String? = null,
    @SerialName("hidden_test_returncode") val hiddenTestReturncode: Int? = null,
    @SerialName("elapsed_s") val elapsedS: Double? = null,
    @SerialName("llm_requests") val llmRequests: Int = 0,
    @SerialName("prompt_tokens") val promptTokens: Int = 0,
    @SerialName("completion_tokens") val completionTokens: Int = 0,
    @SerialName("total_tokens") val totalTokens: Int = 0,
)

@Serializable
data class RunDiagnostic(
    val policy: String,
    @SerialName("bank_id") val bankId: String? = null,
    @SerialName("plan_id") val planId: String? = null,
    @SerialName("plan_signature") val planSignature: String? = null,
    val strategy: String? = null,
    @SerialName("target_files") val targetFiles: List<String> = emptyList(),
    val status: String = "unknown",
    @SerialName("is_best") val isBest: Boolean = false,
    @SerialName("compile_success") val compileSuccess: Boolean? = null,
    @SerialName("compile_error") val compileError: String? = null,
    @SerialName("visible_test_passed") val visibleTestPassed: Boolean? = null,
    @SerialName("visible_test_returncode") val visibleTestReturncode: Int? = null,
    @SerialName("visible_test_stdout") val visibleTestStdout: String? = null,
    @SerialName("visible_test_stderr") val visibleTestStderr: String? = null,
    @SerialName("hidden_test_passed") val hiddenTestPassed: Boolean? = null,
    @SerialName("hidden_test_returncode") val hiddenTestReturncode: Int? = null,
    @SerialName("hidden_test_stdout") val hiddenTestStdout: String? = null,
    @SerialName("hidden_test_stderr") val hiddenTestStderr: String? = null,
)

@Serializable
data class TaskRunResponse(
    val strategy: StrategySummary,
    @SerialName("root_candidates") val rootCandidates: List<Map<String, kotlinx.serialization.json.JsonElement>>,
    @SerialName("plan_bank") val planBank: Map<String, kotlinx.serialization.json.JsonElement> = emptyMap(),
    val nodes: List<Map<String, kotlinx.serialization.json.JsonElement>>,
    val edges: List<List<String>>,
    val transitions: List<Map<String, kotlinx.serialization.json.JsonElement>>,
    @SerialName("best_plan") val bestPlan: Map<String, kotlinx.serialization.json.JsonElement>? = null,
    @SerialName("best_compiled_files") val bestCompiledFiles: Map<String, String> = emptyMap(),
    @SerialName("verifier_summary") val verifierSummary: Map<String, kotlinx.serialization.json.JsonElement> = emptyMap(),
    @SerialName("search_graph_events") val searchGraphEvents: List<Map<String, kotlinx.serialization.json.JsonElement>> = emptyList(),
    val diagnostics: List<RunDiagnostic> = emptyList(),
    @SerialName("run_id") val runId: String? = null,
    @SerialName("cache_key") val cacheKey: String? = null,
    @SerialName("cache_hit") val cacheHit: Boolean = false,
    @SerialName("cloud_status") val cloudStatus: String = "unknown",
)

@Serializable
data class CompareStrategiesResponse(
    val strategies: List<TaskRunResponse>,
    @SerialName("run_id") val runId: String? = null,
    @SerialName("cache_key") val cacheKey: String? = null,
    @SerialName("cache_hit") val cacheHit: Boolean = false,
    @SerialName("cloud_status") val cloudStatus: String = "unknown",
)

@Serializable
data class AsyncJobAcceptedResponse(
    @SerialName("job_id") val jobId: String,
    val kind: String,
)

@Serializable
data class JobProgressSnapshot(
    val phase: String,
    @SerialName("policy") val policy: String? = null,
    @SerialName("current_step") val currentStep: Int = 0,
    @SerialName("max_steps") val maxSteps: Int = 0,
    @SerialName("llm_requests") val llmRequests: Int = 0,
    @SerialName("last_llm_label") val lastLlmLabel: String? = null,
    @SerialName("last_llm_duration_s") val lastLlmDurationS: Double? = null,
    @SerialName("last_compile_duration_s") val lastCompileDurationS: Double? = null,
    @SerialName("last_test_duration_s") val lastTestDurationS: Double? = null,
    @SerialName("compile_successes") val compileSuccesses: Int = 0,
    @SerialName("visible_passes") val visiblePasses: Int = 0,
    val roots: Int? = null,
    val plans: Int? = null,
    val action: String? = null,
    @SerialName("label_tier") val labelTier: String? = null,
    @SerialName("current_policy_index") val currentPolicyIndex: Int = 0,
    @SerialName("total_policies") val totalPolicies: Int = 1,
    @SerialName("elapsed_s") val elapsedS: Double = 0.0,
    val error: String? = null,
)

@Serializable
data class TaskRunJobStatusResponse(
    @SerialName("job_id") val jobId: String,
    val kind: String,
    val status: String,
    val progress: JobProgressSnapshot,
    val result: TaskRunResponse? = null,
)

@Serializable
data class CompareJobStatusResponse(
    @SerialName("job_id") val jobId: String,
    val kind: String,
    val status: String,
    val progress: JobProgressSnapshot,
    val result: CompareStrategiesResponse? = null,
)

@Serializable
data class RunHistoryItem(
    @SerialName("run_id") val runId: String,
    @SerialName("cache_key") val cacheKey: String,
    @SerialName("project_id") val projectId: String,
    @SerialName("project_name") val projectName: String? = null,
    @SerialName("project_root") val projectRoot: String? = null,
    @SerialName("task_root") val taskRoot: String? = null,
    @SerialName("active_file") val activeFile: String? = null,
    @SerialName("prompt_preview") val promptPreview: String,
    val policy: String,
    val family: String,
    @SerialName("created_at") val createdAt: String,
    @SerialName("updated_at") val updatedAt: String,
    @SerialName("visible_passes") val visiblePasses: Int = 0,
    @SerialName("hidden_passes") val hiddenPasses: Int = 0,
    @SerialName("compile_successes") val compileSuccesses: Int = 0,
    @SerialName("total_reward") val totalReward: Double = 0.0,
    @SerialName("elapsed_s") val elapsedS: Double? = null,
    @SerialName("llm_requests") val llmRequests: Int = 0,
    @SerialName("local_status") val localStatus: String = "available",
    @SerialName("cloud_status") val cloudStatus: String = "unknown",
    @SerialName("cloud_object_path") val cloudObjectPath: String? = null,
    @SerialName("package_sha256") val packageSha256: String? = null,
)

@Serializable
data class RunHistoryResponse(
    val items: List<RunHistoryItem>,
    @SerialName("cloud_enabled") val cloudEnabled: Boolean = false,
    val errors: List<String> = emptyList(),
)

@Serializable
data class RunLoadResponse(
    val item: RunHistoryItem,
    val kind: String = "task_run",
    val request: TaskRunRequest? = null,
    val result: TaskRunResponse? = null,
    @SerialName("compare_request") val compareRequest: CompareStrategiesRequest? = null,
    @SerialName("compare_result") val compareResult: CompareStrategiesResponse? = null,
)

@Serializable
data class RunSyncRequest(
    @SerialName("client_context") val clientContext: ClientContext,
    val limit: Int = 100,
)

@Serializable
data class RunSyncResponse(
    @SerialName("cloud_enabled") val cloudEnabled: Boolean,
    val uploaded: Int = 0,
    val downloaded: Int = 0,
    val failed: Int = 0,
    val errors: List<String> = emptyList(),
)
