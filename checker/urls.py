from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("create-checkout-session/", views.create_checkout_session, name="checkout"),
    path("stripe/webhook/", views.stripe_webhook, name="stripe_webhook"),
    path("cancel-subscription/", views.cancel_subscription, name="cancel_subscription"),
    path("settings/", views.settings_view, name="settings"),

]

