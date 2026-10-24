from .main import bp as main_bp
from .videos import bp as videos_bp
from .history import bp as history_bp
from .reports import bp as reports_bp
from .incidents import bp as incidents_bp


def register_routes(app):
    app.register_blueprint(main_bp)
    app.register_blueprint(videos_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(incidents_bp)
