from django.shortcuts import render
from .models import Persons

# Create your views here.


# Home page
def home(request):
    results = Persons.objects.raw("SELECT client_id, current_age, num_cards_issued FROM persons;")
    context = {"results": results}

    return render(request, "fraud_website/pages/home.html", context=context)