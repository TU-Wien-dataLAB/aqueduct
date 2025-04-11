from django import forms
from .models import Team


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
