# Generated migration to convert hair_condition from CharField to JSONField

from django.db import migrations
import json


def convert_hair_condition_to_array(apps, schema_editor):
    """Convert existing hair_condition string values to JSON arrays"""
    UserPreference = apps.get_model('hairmixer_app', 'UserPreference')
    
    for pref in UserPreference.objects.all():
        if pref.hair_condition:
            # If it's already a list (shouldn't be, but safety check)
            if isinstance(pref.hair_condition, list):
                continue
            # Convert string to list with single item
            pref.hair_condition = [pref.hair_condition] if pref.hair_condition else []
        else:
            # Empty string or None → empty list
            pref.hair_condition = []
        pref.save()


def reverse_hair_condition_to_string(apps, schema_editor):
    """Reverse: convert hair_condition arrays back to strings"""
    UserPreference = apps.get_model('hairmixer_app', 'UserPreference')
    
    for pref in UserPreference.objects.all():
        if pref.hair_condition:
            if isinstance(pref.hair_condition, list):
                # Take first item or use empty string
                pref.hair_condition = pref.hair_condition[0] if pref.hair_condition else ''
            # If already string, leave as is
        else:
            pref.hair_condition = ''
        pref.save()


class Migration(migrations.Migration):

    dependencies = [
        ('hairmixer_app', '0005_hairstyle_suitable_gender'),
    ]

    operations = [
        # First, run a data migration to convert existing data
        migrations.RunPython(
            convert_hair_condition_to_array,
            reverse_hair_condition_to_string
        ),
    ]
