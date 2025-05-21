from django import forms
from django.core.exceptions import ValidationError
from django.utils import timezone

from .models import Team, ServiceAccount, Token
from django.conf import settings


class ServiceAccountForm(forms.ModelForm):
    token_expires_at = forms.DateTimeField(
        required=False,
        widget=forms.DateTimeInput(
            attrs={'type': 'datetime-local', 'class': 'input'},
            format='%Y-%m-%dT%H:%M'
        ),
        label='Token Expiration Date (optional)',
        help_text='Set an expiration date for the service account token (optional).'
    )

    class Meta:
        model = ServiceAccount
        fields = ['name', 'description']
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Enter service account name'}),
            'description': forms.Textarea(attrs={'rows': 3, 'placeholder': 'Optional description...'}),
        }
        labels = {
            'name': 'Service Account Name',
            'description': 'Description',
        }

    def __init__(self, *args, **kwargs):
        # Pop the team object passed from the view
        self.team = kwargs.pop('team', None)
        super().__init__(*args, **kwargs)
        # Ensure team was passed if required for validation
        if not self.team:
            # This indicates a programming error in the view setup
            raise ValueError("Team instance must be provided to the ServiceAccountForm.")

    def clean(self):
        cleaned_data = super().clean()
        name = cleaned_data.get('name')

        # Perform the team limit check here
        if self.team:
            # Check for duplicate name within the team
            if name:
                query = ServiceAccount.objects.filter(team=self.team, name__iexact=name)
                instance_pk = self.instance.pk
                if instance_pk:
                    query = query.exclude(pk=instance_pk)
                if query.exists():
                    self.add_error('name', f"A service account with the name '{name}' already exists in team '{self.team.name}'.")

            # Check team limit
            limit = getattr(settings, 'MAX_SERVICE_ACCOUNTS_PER_TEAM', 10)
            query = ServiceAccount.objects.filter(team=self.team)

            # Exclude self if updating (form instance has pk if it's an update)
            instance_pk = self.instance.pk
            if instance_pk:
                query = query.exclude(pk=instance_pk)

            current_count = query.count()

            # Check only needs to prevent *adding* a new one if limit is reached
            # For updates, this check isn't strictly needed unless team can change
            if instance_pk is None and current_count >= limit:
                # Attach the error to the form, not a specific field, as it's a general limit issue
                raise ValidationError(
                    f"Team '{self.team.name}' has reached the maximum limit of {limit} service accounts.",
                    code='limit_reached'
                )
        expires_at = cleaned_data.get('token_expires_at')
        if expires_at:
            if expires_at <= timezone.now():
                self.add_error('expires_at', "Expiration date must be in the future.")
        return cleaned_data


class TeamCreateForm(forms.ModelForm):
    class Meta:
        model = Team
        fields = ['name', 'description']
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Enter team name'}),
            'description': forms.Textarea(attrs={'rows': 3, 'placeholder': 'Optional description...'}),
        }
        labels = {
            'name': 'Team Name',
            'description': 'Description',
        }

    def __init__(self, *args, **kwargs):
        # Pop the organization kwarg before initializing the parent ModelForm
        # We'll receive this from the view
        self.org = kwargs.pop('org', None)
        super().__init__(*args, **kwargs)

    def clean(self):
        """
        Perform validation checks that depend on multiple fields or
        require access to data outside the form fields (like organization).
        """
        cleaned_data = super().clean()
        name = cleaned_data.get('name')

        # We need both the name and the organization (passed during init)
        # to perform the uniqueness check
        if name and self.org:
            # Check if a team with this name already exists in this organization
            query = Team.objects.filter(org=self.org, name__iexact=name)

            # If we are updating an existing instance, exclude it from the check
            if self.instance and self.instance.pk:
                query = query.exclude(pk=self.instance.pk)

            if query.exists():
                # Use __iexact for case-insensitive comparison if desired,
                # or just name=name for case-sensitive.
                self.add_error('name', 'A team with this name already exists in your organization.')

        return cleaned_data


class TokenCreateForm(forms.ModelForm):
    class Meta:
        model = Token
        fields = ['name', 'expires_at']
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Enter token name'}),
            'expires_at': forms.DateTimeInput(
                attrs={'type': 'datetime-local', 'class': 'input'},
                format='%Y-%m-%dT%H:%M'
            ),
        }
        labels = {
            'name': 'Token Name',
            'expires_at': 'Expiration Date (optional)',
        }

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        if not self.user:
            raise ValueError("User must be provided to TokenCreateForm.")
        self.fields['expires_at'].input_formats = ['%Y-%m-%dT%H:%M']

    def clean(self):
        cleaned_data = super().clean()
        from django.conf import settings
        max_tokens = getattr(settings, 'MAX_USER_TOKENS', 3)
        token_count = Token.objects.filter(user=self.user, service_account__isnull=True)
        if self.instance.pk:
            token_count = token_count.exclude(pk=self.instance.pk)
        if token_count.count() >= max_tokens:
            raise ValidationError(f"You can only have {max_tokens} tokens.")
        
        expires_at = cleaned_data.get('expires_at')
        if expires_at:
            if expires_at <= timezone.now():
                self.add_error('expires_at', "Expiration date must be in the future.")

        return cleaned_data
