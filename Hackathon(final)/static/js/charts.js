'{% if ERROR_MSG == "Found" %}'
//var data = JSON.parse('{{ data | tojson | safe}}');
var data = [{
  values: ['data["Positive"]','data["Negative"]'],
  labels: ['positive','negative'],
  marker:{
      colors:['#1ABEE9','#1864BD'],
  },
  title: { text: "Tweets Percentage", font: { size: 24 } },
  type: 'pie'
}];

var layout = {
  height: 400,
  width: 500,
  paper_bgcolor: "#24232A",
};

Plotly.newPlot('graph1', data, layout);


 /** Function to display gauges  */


 var data = JSON.parse('{{ data | tojson | safe}}');
 var pos=data["Positive"];
 var data = [
    {
       type: "indicator",
       mode: "gauge+number+delta",
       value: pos,
title: { text: "Positive Tweets Percentage", font: { size: 24 } },
delta: { reference: 100, increasing: { color: "RebeccaPurple" } },
gauge: {
axis: { range: [null, 100], tickwidth: 1, tickcolor: "#3366cc" },
bar: { color: "#3366CC" },
bgcolor: "#24232A",
borderwidth: 2,
bordercolor: "gray",
steps: [
{ range: [0, 50], color: "cyan" },
{ range: [50, 100], color: "cyan" }
],
threshold: {
line: { color: "red", width: 4 },
thickness: 0.75,
value: 90
}
}
}
];

var layout = {
width: 400,
height: 300,
margin: { t: 25, r: 25, l: 25, b: 25 },
paper_bgcolor: "#24232A",
font: { color: "#3366CC", family: "Arial" }
};

Plotly.newPlot('posgauge', data, layout);


var data = JSON.parse('{{ data | tojson | safe}}');
var neg=data["Negative"];
var data = [
   {
      type: "indicator",
      mode: "gauge+number+delta",
      value: neg,
title: { text: "Negative Tweeets Percentage", font: { size: 24 } },
delta: { reference: 100, increasing: { color: "RebeccaPurple" } },
gauge: {
axis: { range: [null, 100], tickwidth: 1, tickcolor: "#3366cc" },
bar: { color: "#3366CC" },
bgcolor: "white",
borderwidth: 2,
bordercolor: "gray",
steps: [
{ range: [0, 50], color: "cyan" },
{ range: [50, 100], color: "cyan" }
],
threshold: {
line: { color: "red", width: 4 },
thickness: 0.75,
value: 90
}
}
}
];

var layout = {
width: 400,
height: 300,
margin: { t: 25, r: 25, l: 25, b: 25 },
paper_bgcolor: "#24232A",
font: { color: "#3366CC", family: "Arial" }
};

Plotly.newPlot('neggauge', data, layout);

    
             let values=[]
             var count=0
             let freq=[]
            '{% for key,values in words.items() %}'
                
                    if(count!=30){
                       values.push('{{values}}')
                       count=count+1;
                    }
            '{% endfor %}'
            count=0
            '{% for key,values in freq.items() %}'
                
                if(count!=30){
                   freq.push('{{values}}')
                   count=count+1;
                }
        '{% endfor %}'

            console.log(values);
            console.log(freq)
            $(document).ready(function(){
                var entries= values.map(x => ({label:x,url:"https://twitter.com/search?q="+x}));
                var settings = {
                    entries :entries,
                    width:640,
                    height:400,
                    radius:'65%',
                    radiusMin:85,
                    bgDraw:true,
                    bgColor:"#24232A",
                    opacityOver:1.00,
                    opacitySpeed:6,
                    fov:800,
                    speed:2,
                    fontFamily:'Courier,Arial,sans-serif',
                    fontSize:'30',
                    fontColor:'cyan',
                    fontWeight:'bold',
                    fontStyle:'normal',
                    fontToUppercase:true,
                };
                $('#graph3').svg3DTagCloud(settings)
            });
           
            var data = [
                  {
                     x: values,
                     y: freq,
                     marker:{
                       color: ["#FFA07A","#FA8072","#E9967A","#F08080","#FF7F50"
                       ,"#FF6347","#FFFFE0","#FFFACD","#FAFAD2","#FFEFD5"
                       ,"#FFE4B5","#FFDAB9","#EEE8AA","#F0E68C","#BDB76B"
                       ,"#FFFF00","#E0FFFF","#00FFFF","#00FFFF","#7FFFD4","66CDAA"
                       ,"#AFEEEE","#40E0D0","#48D1CC","#00CED1"
                       ,"#20B2AA","#5F9EA0"
                       ,"#008B8B","#008080"]
                      },
                     type: 'bar',
                     bgcolor: "black",
                     title: { text: "word freuqency", font: { size: 24 } },
                  }
              ];
            
              var layout = {
                      width: 800,
                      height: 400,
                      paper_bgcolor: "#24232A",
                      
                      font: { color: "white", family: "Arial" }
                  };
      Plotly.newPlot('graph4', data,layout);
      '{%endif%}'