defmodule Driftscape.Repo do
  use Ecto.Repo,
    otp_app: :driftscape,
    adapter: Ecto.Adapters.SQLite3
end
