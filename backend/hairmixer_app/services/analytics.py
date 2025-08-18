import logging
from django.utils import timezone
from django.db import transaction
from typing import Dict, Any, Optional
from ..models import AnalyticsEvent, CustomUser

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Centralized analytics service for tracking user interactions"""
    
    def __init__(self):
        self.enabled = True
    
    def track_event(
        self, 
        user: Optional[CustomUser],
        event_type: str,
        event_data: Dict[str, Any] = None,
        session_id: str = '',
        request=None
    ):
        """Track an analytics event"""
        try:
            if not self.enabled:
                return
            
            # Extract request information
            ip_address = None
            user_agent = ''
            
            if request:
                ip_address = self._get_client_ip(request)
                user_agent = request.META.get('HTTP_USER_AGENT', '')[:500]  # Limit length
            
            # Create analytics event
            with transaction.atomic():
                AnalyticsEvent.objects.create(
                    user=user,
                    event_type=event_type,
                    event_data=event_data or {},
                    session_id=session_id[:100],  # Limit length
                    ip_address=ip_address,
                    user_agent=user_agent
                )
            
            logger.info(f"Analytics event tracked: {event_type} for user {user.id if user else 'anonymous'}")
            
        except Exception as e:
            logger.error(f"Error tracking analytics event: {str(e)}")
            # Don't raise exception to avoid breaking main functionality
    
    def track_page_view(self, user: Optional[CustomUser], page_name: str, request=None):
        """Track page view events"""
        self.track_event(
            user=user,
            event_type='page_view',
            event_data={'page': page_name},
            request=request
        )
    
    def track_user_journey(self, user: Optional[CustomUser], journey_step: str, metadata: Dict = None, request=None):
        """Track user journey progression"""
        self.track_event(
            user=user,
            event_type='user_journey',
            event_data={
                'step': journey_step,
                'metadata': metadata or {}
            },
            request=request
        )
    
    def track_performance_metric(self, metric_name: str, value: float, metadata: Dict = None):
        """Track performance metrics"""
        self.track_event(
            user=None,
            event_type='performance_metric',
            event_data={
                'metric': metric_name,
                'value': value,
                'metadata': metadata or {}
            }
        )
    
    def track_error(self, error_type: str, error_message: str, user: Optional[CustomUser] = None, request=None):
        """Track error events"""
        self.track_event(
            user=user,
            event_type='error_occurred',
            event_data={
                'error_type': error_type,
                'error_message': error_message[:1000]  # Limit length
            },
            request=request
        )
    
    def _get_client_ip(self, request):
        """Extract client IP from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
    
    def get_user_analytics(self, user: CustomUser, days: int = 30):
        """Get analytics summary for a user"""
        try:
            from django.db.models import Count
            from datetime import timedelta
            
            start_date = timezone.now() - timedelta(days=days)
            
            events = AnalyticsEvent.objects.filter(
                user=user,
                created_at__gte=start_date
            ).values('event_type').annotate(
                count=Count('id')
            ).order_by('-count')
            
            return {
                'user_id': user.id,
                'period_days': days,
                'total_events': sum(event['count'] for event in events),
                'event_breakdown': list(events),
                'generated_at': timezone.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating user analytics: {str(e)}")
            return {}
    
    def get_system_analytics(self, days: int = 7):
        """Get system-wide analytics"""
        try:
            from django.db.models import Count, Avg
            from datetime import timedelta
            
            start_date = timezone.now() - timedelta(days=days)
            
            # Event counts by type
            event_counts = AnalyticsEvent.objects.filter(
                created_at__gte=start_date
            ).values('event_type').annotate(
                count=Count('id')
            ).order_by('-count')
            
            # User activity
            active_users = AnalyticsEvent.objects.filter(
                created_at__gte=start_date,
                user__isnull=False
            ).values('user').distinct().count()
            
            # Performance metrics
            perf_metrics = AnalyticsEvent.objects.filter(
                event_type='performance_metric',
                created_at__gte=start_date
            ).values('event_data__metric').annotate(
                avg_value=Avg('event_data__value'),
                count=Count('id')
            )
            
            return {
                'period_days': days,
                'total_events': sum(event['count'] for event in event_counts),
                'event_breakdown': list(event_counts),
                'active_users': active_users,
                'performance_metrics': list(perf_metrics),
                'generated_at': timezone.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating system analytics: {str(e)}")
            return {}