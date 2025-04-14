from django import forms
from django.core.exceptions import ValidationError

from .models import Team, ServiceAccount
from django.conf import settings


class ServiceAccountForm(forms.ModelForm):
    class Meta:
        model = ServiceAccount
        fields = ['name']
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Enter service account name'}),
        }
        labels = {
            'name': 'Service Account Name',
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

        # Perform the team limit check here
        if self.team:
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
                raise ValidationError(
                    f"Team '{self.team.name}' has reached the maximum limit of {limit} service accounts.",
                    code='limit_reached'  # Optional error code
                )

        return cleaned_data


class TeamCreateForm(forms.ModelForm):
    class Meta:
        model = Team
        fields = ['name']  # We only need the user to input the name

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
            if Team.objects.filter(org=self.org, name__iexact=name).exists():
                # Use __iexact for case-insensitive comparison if desired,
                # or just name=name for case-sensitive.
                self.add_error('name', 'A team with this name already exists in your organization.')
                # Note: Using add_error('name', ...) attaches the error specifically
                #  to the 'name' field in the form.

        return cleaned_data
