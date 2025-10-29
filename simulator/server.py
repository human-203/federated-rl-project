# simulator/server.py
import numpy as np

class Server:
    def __init__(self):
        self.global_model_version = 0
        self.history = []

    def aggregate(self, client_updates):
        # simple average aggregator (placeholder)
        # client_updates: list of (weights, metadata)
        if not client_updates:
            return None
        # here we only simulate, so update version and record metrics
        self.global_model_version += 1
        self.history.append({"version": self.global_model_version, "n_clients": len(client_updates)})
        return self.global_model_version
