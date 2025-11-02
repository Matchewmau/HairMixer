# Generated migration for PreferenceProfile model updates

from django.db import migrations, models


def convert_hair_condition_to_list(apps, schema_editor):
    """Convert existing hair_condition CharField values to JSON arrays"""
    PreferenceProfile = apps.get_model('hairmixer_app', 'PreferenceProfile')
    for profile in PreferenceProfile.objects.all():
        if profile.hair_condition and isinstance(profile.hair_condition, str):
            # Convert single value to list
            profile.hair_condition = [profile.hair_condition]
            profile.save()


class Migration(migrations.Migration):

    dependencies = [
        ('hairmixer_app', '0008_preferenceprofile'),
    ]

    operations = [
        # Step 1: Rename maintenance_level to maintenance
        migrations.RenameField(
            model_name='preferenceprofile',
            old_name='maintenance_level',
            new_name='maintenance',
        ),
        
        # Step 2: Add new fields with defaults
        migrations.AddField(
            model_name='preferenceprofile',
            name='hair_color',
            field=models.CharField(
                choices=[
                    ('black', 'Black'), ('brown', 'Brown'), ('blonde', 'Blonde'),
                    ('red', 'Red'), ('auburn', 'Auburn'), ('gray', 'Gray'),
                    ('white', 'White'), ('other', 'Other')
                ],
                default='brown',
                max_length=20
            ),
        ),
        migrations.AddField(
            model_name='preferenceprofile',
            name='hair_texture_detail',
            field=models.CharField(
                choices=[
                    ('fine', 'Fine'), ('normal', 'Normal'), ('thick', 'Thick'),
                    ('smooth', 'Smooth'), ('coarse', 'Coarse'), ('silky', 'Silky'),
                    ('frizzy', 'Frizzy')
                ],
                default='normal',
                max_length=20
            ),
        ),
        migrations.AddField(
            model_name='preferenceprofile',
            name='hair_thickness',
            field=models.CharField(
                choices=[
                    ('thin', 'Thin'), ('medium', 'Medium'),
                    ('thick', 'Thick'), ('very_thick', 'Very Thick')
                ],
                default='medium',
                max_length=20
            ),
        ),
        migrations.AddField(
            model_name='preferenceprofile',
            name='styling_preference',
            field=models.CharField(
                choices=[
                    ('natural', 'Natural'), ('casual', 'Casual'), ('classic', 'Classic'),
                    ('polished', 'Polished'), ('elegant', 'Elegant'),
                    ('glamorous', 'Glamorous'), ('trendy', 'Trendy'), ('edgy', 'Edgy')
                ],
                default='natural',
                max_length=20
            ),
        ),
        migrations.AddField(
            model_name='preferenceprofile',
            name='volume',
            field=models.CharField(
                choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High')],
                default='medium',
                max_length=20
            ),
        ),
        
        # Step 3: Rename hair_condition temporarily to preserve data
        migrations.RenameField(
            model_name='preferenceprofile',
            old_name='hair_condition',
            new_name='hair_condition_old',
        ),
        
        # Step 4: Add new JSONField for hair_condition
        migrations.AddField(
            model_name='preferenceprofile',
            name='hair_condition',
            field=models.JSONField(
                blank=True,
                default=list,
                help_text='Multiple hair conditions'
            ),
        ),
        
        # Step 5: Run data migration to convert old values to new format
        migrations.RunPython(convert_hair_condition_to_list, migrations.RunPython.noop),
        
        # Step 6: Remove old field and unused fields
        migrations.RemoveField(
            model_name='preferenceprofile',
            name='hair_condition_old',
        ),
        migrations.RemoveField(
            model_name='preferenceprofile',
            name='avoid_styles',
        ),
        migrations.RemoveField(
            model_name='preferenceprofile',
            name='hair_texture',
        ),
        migrations.RemoveField(
            model_name='preferenceprofile',
            name='preferred_colors',
        ),
        
        # Step 7: Alter other fields
        migrations.AlterField(
            model_name='preferenceprofile',
            name='gender',
            field=models.CharField(
                choices=[('male', 'Male'), ('female', 'Female')],
                default='female',
                max_length=20
            ),
        ),
        migrations.AlterField(
            model_name='preferenceprofile',
            name='hair_length',
            field=models.CharField(
                choices=[('short', 'Short'), ('medium', 'Medium'), ('long', 'Long')],
                default='medium',
                max_length=20
            ),
        ),
        migrations.AlterField(
            model_name='preferenceprofile',
            name='lifestyle',
            field=models.CharField(
                choices=[
                    ('active', 'Active'), ('professional', 'Professional'),
                    ('creative', 'Creative'), ('casual', 'Casual'),
                    ('moderate', 'Moderate'), ('relaxed', 'Relaxed')
                ],
                default='casual',
                max_length=20
            ),
        ),
    ]
