from app import create_app

app = create_app()

if __name__ == "__main__":
    # Keep development startup to one process; the reloader can otherwise run
    # competing background-worker recovery during project demonstrations.
    app.run(debug=app.config["DEBUG"], use_reloader=False)
