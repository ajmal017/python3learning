/**
 * Created by kyoka on 2017/12/10.
 */



function selfplay() {
    //alert("start selfplay");
    var isContinue = true;
    var count = 0;
    // while (isContinue) {
    	var req = {url: "/nextMove/kyoka",
					asyn: true,
					success: function(ret){
    					ret = JSON.parse(ret);
						var isEnd = ret["end"];
						if (isEnd == true) {
							isContinue = false;
						} else {
							var x_ = parseInt(ret["row"]);
							var y_ = parseInt(ret["col"]);
							play(x_, y_);
							showPan();
						}
					}};
        $.ajax(req);
		// setTimeout(function(){},1000);
    // }
}