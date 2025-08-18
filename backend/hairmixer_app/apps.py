from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class HairmixerAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'hairmixer_app'
    verbose_name = 'HairMixer Application'
    
    def ready(self):
        """Initialize app when Django starts"""
        try:
            # Import signals if you have any
            # import hairmixer_app.signals
            
            # Initialize ML models
            # self.initialize_ml_models()
            
            # # Set up periodic tasks
            # self.setup_periodic_tasks()
            
            logger.info("HairMixer app initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing HairMixer app: {str(e)}")
    
    def initialize_ml_models(self):
        """Initialize ML models on app startup"""
        # try:
        #     from .ml.model import load_model
        #     # Pre-load the model to avoid cold starts
        #     load_model()
        # except Exception as e:
        #     logger.error(f"Error initializing ML models: {str(e)}")
        pass
    
    def setup_periodic_tasks(self):
        """Set up periodic tasks (cache cleanup, analytics, etc.)"""
        try:
            # TODO: Set up Celery tasks or Django-cron jobs
            # For now, we'll rely on management commands
            pass
        except Exception as e:
            logger.error(f"Error setting up periodic tasks: {str(e)}")
