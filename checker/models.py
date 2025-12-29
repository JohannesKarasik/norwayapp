from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    is_paying = models.BooleanField(default=False)
    stripe_customer_id = models.CharField(max_length=255, blank=True, null=True)

    # ✅ ADD THIS LINE
    stripe_subscription_id = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self) -> str:
        return f"Profile({self.user.username})"



@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    # Safe no-op if profile doesn't exist yet; ensures updates won't error
    try:
        instance.profile.save()
    except Profile.DoesNotExist:
        pass
