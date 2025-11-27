from django.db import migrations


def convert_hair_condition_to_array(apps, schema_editor):
    """Convert existing hair_condition string values to JSON arrays"""
    UserPreference = apps.get_model('hairmixer_app', 'UserPreference')
    
    for pref in UserPreference.objects.all():
        if pref.hair_condition:
            if isinstance(pref.hair_condition, list):
                continue
            pref.hair_condition = (
                [pref.hair_condition] if pref.hair_condition else []
            )
        else:
            pref.hair_condition = []
        pref.save()


def reverse_hair_condition_to_string(apps, schema_editor):
    """Reverse: convert hair_condition arrays back to strings"""
    UserPreference = apps.get_model('hairmixer_app', 'UserPreference')
    
    for pref in UserPreference.objects.all():
        if pref.hair_condition:
            if isinstance(pref.hair_condition, list):
                pref.hair_condition = (
                    pref.hair_condition[0] if pref.hair_condition else ''
                )
        else:
            pref.hair_condition = ''
        pref.save()


class Migration(migrations.Migration):

    dependencies = [
        ('hairmixer_app', '0005_hairstyle_suitable_gender'),
    ]

    operations = [
        migrations.RunPython(
            convert_hair_condition_to_array,
            reverse_hair_condition_to_string
        ),
    ]
