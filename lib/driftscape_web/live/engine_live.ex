defmodule DriftscapeWeb.EngineLive do
  use DriftscapeWeb, :live_view

  # Ensure these atoms exist for String.to_existing_atom/1
  @model_statuses [:loading, :downloading, :ready, :error]
  def model_statuses, do: @model_statuses

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     assign(socket,
       prompt: "",
       model_status: :loading,
       model_detail: "Preparing AI engine...",
       toast_visible: true,
       generating: false,
       gen_progress: 0,
       gen_label: "",
       playing: false
     )}
  end

  @impl true
  def handle_event("update-prompt", %{"value" => value}, socket) do
    {:noreply, assign(socket, prompt: value)}
  end

  def handle_event("generate", _params, socket) do
    socket =
      socket
      |> assign(generating: true, gen_progress: 0, gen_label: "Starting...")
      |> push_event("generate-audio", %{prompt: socket.assigns.prompt})

    {:noreply, socket}
  end

  def handle_event("model-status", %{"status" => status, "detail" => detail}, socket) do
    model_status = String.to_existing_atom(status)

    socket = assign(socket, model_status: model_status, model_detail: detail, toast_visible: true)

    if model_status == :ready do
      Process.send_after(self(), :dismiss_toast, 4000)
    end

    {:noreply, socket}
  end

  def handle_event("dismiss-toast", _params, socket) do
    {:noreply, assign(socket, toast_visible: false)}
  end

  def handle_event("generation-progress", %{"progress" => progress, "label" => label}, socket) do
    {:noreply, assign(socket, gen_progress: progress, gen_label: label)}
  end

  def handle_event("generation-complete", _params, socket) do
    {:noreply, assign(socket, generating: false, playing: true, gen_progress: 0, gen_label: "")}
  end

  def handle_event("stop", _params, socket) do
    {:noreply,
     socket
     |> assign(playing: false)
     |> push_event("stop-audio", %{})}
  end

  @impl true
  def handle_info(:dismiss_toast, socket) do
    {:noreply, assign(socket, toast_visible: false)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="min-h-screen bg-base-200 flex flex-col items-center justify-center p-8">
      <div class="max-w-2xl w-full space-y-8">
        <div class="text-center">
          <h1 class="text-4xl font-bold">Driftscape</h1>
          <p class="text-base-content/60 mt-2">Describe a mood. Let AI compose the soundscape.</p>
        </div>

        <div id="audio-engine" phx-hook="AudioEngine" phx-update="ignore"></div>

        <div class="card bg-base-100 shadow-xl">
          <div class="card-body space-y-4">
            <div class="form-control">
              <label class="label">
                <span class="label-text text-base font-medium">What does your moment sound like?</span>
              </label>
              <input
                type="text"
                placeholder="e.g. gentle rain on a forest canopy at dusk"
                class="input input-bordered input-lg w-full"
                value={@prompt}
                phx-keyup="update-prompt"
                disabled={@generating}
              />
            </div>

            <button
              class={[
                "btn btn-primary btn-lg w-full",
                (@generating || not can_generate?(assigns)) && "btn-disabled"
              ]}
              phx-click="generate"
              disabled={not can_generate?(assigns) || @generating}
            >
              <%= if @generating do %>
                <span class="loading loading-spinner"></span>
                Generating...
              <% else %>
                <%= if @playing, do: "Generate New", else: "Generate" %>
              <% end %>
            </button>

            <div :if={@generating} class="space-y-2">
              <div class="flex justify-between text-sm text-base-content/70">
                <span>{@gen_label}</span>
                <span>{@gen_progress}%</span>
              </div>
              <progress class="progress progress-primary w-full" value={@gen_progress} max="100">
              </progress>
            </div>

            <div :if={@playing && !@generating} class="flex items-center justify-center gap-4 pt-2">
              <button class="btn btn-circle btn-outline" phx-click="stop">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  class="h-5 w-5"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <rect x="6" y="6" width="12" height="12" rx="1" />
                </svg>
              </button>
              <span class="text-sm text-base-content/60">Looping ambient soundscape</span>
            </div>
          </div>
        </div>
      </div>

      <%!-- Toast / snackbar for model status --%>
      <div :if={@toast_visible} class="fixed bottom-6 left-1/2 -translate-x-1/2 z-50">
        <div class={[
          "flex items-center gap-3 px-5 py-3 rounded-xl shadow-lg text-sm transition-all",
          toast_classes(@model_status)
        ]}>
          <span :if={@model_status != :ready} class="loading loading-spinner loading-sm"></span>
          <svg
            :if={@model_status == :ready}
            xmlns="http://www.w3.org/2000/svg"
            class="h-5 w-5"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fill-rule="evenodd"
              d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
              clip-rule="evenodd"
            />
          </svg>
          <span>{toast_text(@model_status, @model_detail)}</span>
          <button
            :if={@model_status == :ready}
            class="ml-2 opacity-60 hover:opacity-100"
            phx-click="dismiss-toast"
          >
            &times;
          </button>
        </div>
      </div>
    </div>
    """
  end

  defp can_generate?(assigns) do
    assigns.prompt != "" and assigns.model_status == :ready
  end

  defp toast_classes(:ready), do: "bg-success text-success-content"
  defp toast_classes(_), do: "bg-base-300 text-base-content"

  defp toast_text(:ready, _detail), do: "AI engine ready"
  defp toast_text(_status, detail), do: detail
end
