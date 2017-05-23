/**
 * Created by Bk120 on 2017-05-22.
 */

function main() {
    $('.details').hide();
    $('.detail_btn').on('click', function() {
       $(this).prev().slideToggle(400);
       $(this).toggleClass('active');
    });
}

$(document).ready(main);