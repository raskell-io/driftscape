defmodule DriftscapeWeb.PageController do
  use DriftscapeWeb, :controller

  def home(conn, _params) do
    render(conn, :home)
  end
end
