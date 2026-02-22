defmodule DriftscapeWeb.EngineLive do
  use DriftscapeWeb, :live_view

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     assign(socket,
       prompt: "",
       status: :idle,
       progress: 0,
       progress_label: ""
     )}
  end

  @impl true
  def handle_event("update-prompt", %{"prompt" => prompt}, socket) do
    {:noreply, assign(socket, prompt: prompt)}
  end

  def handle_event("generate", _params, socket) do
    {:noreply, assign(socket, status: :loading, progress: 0, progress_label: "Loading model...")}
  end

  def handle_event("generation-progress", %{"progress" => progress, "label" => label}, socket) do
    {:noreply, assign(socket, progress: progress, progress_label: label)}
  end

  def handle_event("generation-complete", _params, socket) do
    {:noreply, assign(socket, status: :playing, progress: 100, progress_label: "Playing")}
  end

  def handle_event("playback-stopped", _params, socket) do
    {:noreply, assign(socket, status: :idle)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="min-h-screen bg-base-200 flex flex-col items-center justify-center p-8">
      <div class="max-w-2xl w-full space-y-8">
        <div class="text-center">
          <h1 class="text-4xl font-bold">Driftscape</h1>
          <p class="text-base-content/60 mt-2">AI-powered ambient soundscapes</p>
        </div>

        <div id="audio-engine" phx-hook="AudioEngine" phx-update="ignore" data-prompt={@prompt}>
          <!-- Web Audio API context managed by JS hook -->
        </div>

        <div class="card bg-base-100 shadow-xl">
          <div class="card-body space-y-4">
            <div class="form-control">
              <label class="label">
                <span class="label-text">Describe your soundscape</span>
              </label>
              <input
                type="text"
                placeholder="e.g. gentle rain on a forest canopy at dusk"
                class="input input-bordered w-full"
                value={@prompt}
                phx-keyup="update-prompt"
                phx-key=""
                name="prompt"
                disabled={@status in [:loading, :generating]}
              />
            </div>

            <button
              class={[
                "btn btn-primary w-full",
                @status in [:loading, :generating] && "btn-disabled"
              ]}
              phx-click="generate"
              disabled={@prompt == "" or @status in [:loading, :generating]}
            >
              <span :if={@status in [:loading, :generating]} class="loading loading-spinner"></span>
              {generate_button_text(@status)}
            </button>

            <div :if={@status != :idle} class="space-y-2">
              <div class="flex justify-between text-sm">
                <span>{@progress_label}</span>
                <span>{@progress}%</span>
              </div>
              <progress class="progress progress-primary w-full" value={@progress} max="100">
              </progress>
            </div>

            <div :if={@status == :playing} class="flex items-center justify-center gap-4 pt-2">
              <button class="btn btn-circle btn-outline" phx-click="playback-stopped">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  class="h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <rect x="6" y="4" width="4" height="16" />
                  <rect x="14" y="4" width="4" height="16" />
                </svg>
              </button>
              <span class="text-sm text-base-content/60">Ambient loop playing</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    """
  end

  defp generate_button_text(:idle), do: "Generate"
  defp generate_button_text(:loading), do: "Loading Model..."
  defp generate_button_text(:generating), do: "Generating..."
  defp generate_button_text(:playing), do: "Generate New"
end
