from ratelimit import limits, sleep_and_retry
import requests

class RateLimiter:
    @sleep_and_retry
    @limits(calls=30, period=60)  # Adjust the limits as needed
    def limited_request(self, method, url, **kwargs):
        return requests.request(method, url, **kwargs)
