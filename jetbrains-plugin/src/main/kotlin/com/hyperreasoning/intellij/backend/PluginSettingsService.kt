package com.hyperreasoning.intellij.backend

import com.intellij.openapi.components.PersistentStateComponent
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.State
import com.intellij.openapi.components.Storage

@Service(Service.Level.APP)
@State(name = "HyperreasoningPluginSettings", storages = [Storage("hyperreasoning-plugin.xml")])
class PluginSettingsService : PersistentStateComponent<PluginSettingsService.State> {
    data class State(
        var backendBaseUrl: String = "http://127.0.0.1:8765",
        var checkpointPath: String = "",
        var proposalSource: String = "heuristic",
        var runTests: Boolean = true,
        var runHiddenTests: Boolean = false,
    )

    private var state = State()

    override fun getState(): State = state

    override fun loadState(state: State) {
        this.state = state
    }
}
