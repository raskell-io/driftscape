defmodule DriftscapeWeb.Plugs.CrossOriginIsolation do
  @moduledoc """
  Sets Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers
  required for SharedArrayBuffer access (used by AudioWorklet ↔ Worker communication).
  """

  import Plug.Conn

  def init(opts), do: opts

  def call(conn, _opts) do
    conn
    |> put_resp_header("cross-origin-opener-policy", "same-origin")
    |> put_resp_header("cross-origin-embedder-policy", "require-corp")
  end
end
