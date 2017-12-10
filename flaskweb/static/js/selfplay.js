/**
 * Created by kyoka on 2017/12/10.
 */


function selfplay() {
    alert("start selfplay");
    var isContinue = true;
    var count = 0;
    while (isContinue) {
        $.get("/nextMove/", {}, function(ret){
			ret = JSON.parse(ret);
			var isEnd = ret["end"];
			if (isEnd == true) {
				isContinue = false;
			} else {
				var x_ = ret["row"];
				var y_ = ret["col"];
				play(x_, y_);
				showPan();
			}
		})
    }
}