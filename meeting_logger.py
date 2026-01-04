import json
import os
from datetime import datetime
import pandas as pd

class MeetingLogger:
    """Structured logging for meeting sessions"""

    def __init__(self, log_dir="meeting_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def create_session_id(self):
        """Generate unique session ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_session(self, transcript, summary, segments, metadata=None):
        """Save complete meeting session"""
        session_id = self.create_session_id()

        # Prepare session data
        session_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "transcript": {
                "full_text": transcript,
                "length": len(transcript),
                "word_count": len(transcript.split())
            },
            "summary": {
                "text": summary,
                "length": len(summary)
            },
            "speakers": {
                "count": len(set([s['speaker'] for s in segments])),
                "segments": segments
            },
            "metrics": metadata.get('metrics', {}) if metadata else {}
        }

        # Save as JSON
        json_path = os.path.join(self.log_dir, f"{session_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        # Save as Parquet for analytics
        try:
            parquet_path = os.path.join(self.log_dir, f"{session_id}.parquet")

            # Flatten data for parquet
            df_data = {
                'session_id': [session_id],
                'timestamp': [session_data['timestamp']],
                'transcript': [transcript],
                'summary': [summary],
                'num_speakers': [session_data['speakers']['count']],
                'word_count': [session_data['transcript']['word_count']],
                'duration': [metadata.get('duration', 0) if metadata else 0],
                'model': [metadata.get('model', 'unknown') if metadata else 'unknown']
            }

            df = pd.DataFrame(df_data)
            df.to_parquet(parquet_path, index=False)

        except Exception as e:
            print(f"Parquet save error: {e}")

        return session_id, json_path

    def load_session(self, session_id):
        """Load a specific session"""
        json_path = os.path.join(self.log_dir, f"{session_id}.json")

        if not os.path.exists(json_path):
            return None

        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_sessions(self):
        """List all saved sessions"""
        sessions = []

        for filename in os.listdir(self.log_dir):
            if filename.endswith('.json'):
                session_id = filename.replace('.json', '')
                json_path = os.path.join(self.log_dir, filename)

                with open(json_path, 'r') as f:
                    data = json.load(f)
                    sessions.append({
                        'session_id': session_id,
                        'timestamp': data.get('timestamp'),
                        'speakers': data.get('speakers', {}).get('count', 0),
                        'word_count': data.get('transcript', {}).get('word_count', 0)
                    })

        return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)

    def get_analytics(self):
        """Get analytics across all sessions"""
        try:
            parquet_files = [f for f in os.listdir(self.log_dir) if f.endswith('.parquet')]

            if not parquet_files:
                return None

            dfs = []
            for pf in parquet_files:
                df = pd.read_parquet(os.path.join(self.log_dir, pf))
                dfs.append(df)

            combined_df = pd.concat(dfs, ignore_index=True)

            analytics = {
                'total_meetings': len(combined_df),
                'total_words': combined_df['word_count'].sum(),
                'avg_duration': combined_df['duration'].mean(),
                'avg_speakers': combined_df['num_speakers'].mean(),
                'models_used': combined_df['model'].value_counts().to_dict()
            }

            return analytics

        except Exception as e:
            print(f"Analytics error: {e}")
            return None

    def export_all_sessions(self, format='csv'):
        """Export all sessions metadata"""
        sessions = self.list_sessions()

        if format == 'csv':
            df = pd.DataFrame(sessions)
            csv_path = os.path.join(self.log_dir, 'all_sessions.csv')
            df.to_csv(csv_path, index=False)
            return csv_path

        elif format == 'json':
            json_path = os.path.join(self.log_dir, 'all_sessions.json')
            with open(json_path, 'w') as f:
                json.dump(sessions, f, indent=2)
            return json_path

        return None
