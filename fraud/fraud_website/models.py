# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class Cards(models.Model):
    mcc_description = models.TextField(blank=True, null=True)
    transaction_id = models.BigIntegerField(blank=True, null=True)
    date = models.TextField(blank=True, null=True)
    client_id = models.BigIntegerField(blank=True, null=True)
    card_id = models.BigIntegerField(blank=True, null=True)
    amount = models.FloatField(blank=True, null=True)
    use_chip = models.TextField(blank=True, null=True)
    merchant_id = models.BigIntegerField(blank=True, null=True)
    merchant_city = models.TextField(blank=True, null=True)
    merchant_state = models.TextField(blank=True, null=True)
    credit_score = models.BigIntegerField(blank=True, null=True)
    card_brand = models.TextField(blank=True, null=True)
    card_type = models.TextField(blank=True, null=True)
    card_number = models.BigIntegerField(blank=True, null=True)
    expires = models.TextField(blank=True, null=True)
    cvv = models.BigIntegerField(blank=True, null=True)
    has_chip = models.TextField(blank=True, null=True)
    credit_limit = models.FloatField(blank=True, null=True)
    card_on_dark_web = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cards'


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class Persons(models.Model):
    client_id = models.BigIntegerField(primary_key=True)
    gender = models.TextField(blank=True, null=True)
    current_age = models.BigIntegerField(blank=True, null=True)
    retirement_age = models.BigIntegerField(blank=True, null=True)
    birth_month = models.BigIntegerField(blank=True, null=True)
    birth_year = models.BigIntegerField(blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    per_capita_income = models.FloatField(blank=True, null=True)
    yearly_income = models.FloatField(blank=True, null=True)
    total_debt = models.FloatField(blank=True, null=True)
    num_cards_issued = models.BigIntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'persons'
