import argparse
import logging
from pathlib import Path
import time
import json
from eeg_stream.eeg_processor import EEGProcessor
from avatar_display.avatar_processor import AvatarProcessor

class FeedbackSystem:
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.initialize_components()

    def _load_config(self, config_path):
        default_config = {
            'eeg_config_path': 'config/eeg_config.json',
            'avatar_config_path': 'config/avatar_config.json',
            'feedback_interval': 2.0,  # seconds
            'session_duration': 300,  # seconds
            'video_path': 'assets/default_avatar.mp4',
            'save_metrics': True,
            'metrics_path': 'metrics/'
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        
        return default_config

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('feedback_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FeedbackSystem')

    def initialize_components(self):
        """Initialize EEG and Avatar processors"""
        try:
            self.eeg_processor = EEGProcessor(self.config['eeg_config_path'])
            self.avatar_processor = AvatarProcessor(self.config['avatar_config_path'])
            self.logger.info("Components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def run_feedback_loop(self):
        """Main feedback loop"""
        try:
            session_start = time.time()
            metrics = []
            
            self.logger.info("Starting feedback loop...")
            
            while time.time() - session_start < self.config['session_duration']:
                loop_start = time.time()
                
                # Get cognitive state from EEG
                cognitive_state = self.eeg_processor.get_cognitive_state()
                
                # Process avatar with cognitive state
                processed_video = self.avatar_processor.process_video(
                    self.config['video_path'],
                    cognitive_state
                )
                
                # Save metrics if enabled
                if self.config['save_metrics']:
                    metrics.append({
                        'timestamp': time.time(),
                        'cognitive_state': cognitive_state
                    })
                
                # Wait for next interval
                processing_time = time.time() - loop_start
                wait_time = max(0, self.config['feedback_interval'] - processing_time)
                time.sleep(wait_time)
                
            # Save session metrics
            if self.config['save_metrics']:
                self._save_metrics(metrics)
                
            self.logger.info("Feedback loop completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in feedback loop: {str(e)}")
            raise
        finally:
            self.cleanup()

    def _save_metrics(self, metrics):
        """Save session metrics to file"""
        try:
            metrics_dir = Path(self.config['metrics_path'])
            metrics_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            metrics_file = metrics_dir / f"session_metrics_{timestamp}.json"
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            self.logger.info(f"Metrics saved to {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {str(e)}")

    def cleanup(self):
        """Clean up resources"""
        try:
            self.eeg_processor.cleanup()
            self.avatar_processor.cleanup()
            self.logger.info("Resources cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='EEG-Avatar Feedback System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        system = FeedbackSystem(args.config)
        system.run_feedback_loop()
    except Exception as e:
        logging.error(f"System error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
