const ctx=document.getElementById("chart");

new Chart(ctx,{

type:"doughnut",

data:{

labels:["Real","Fake"],

datasets:[{

data:[40,60]

}]

}

});