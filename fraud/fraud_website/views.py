from django.shortcuts import render
from .models import Fraud1

# Create your views here.


# Home page
def home(request):
    results = Fraud1.objects.raw("SELECT DISTINCT(client_id_x), card_number, card_brand, card_type FROM fraud1;")
    context = {"results": results}

    return render(request, "fraud_website/pages/home.html", context=context)