$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    // Add 'aws-amplify' library into your application

// Configure Auth category with your Amazon Cognito credentials
    Amplify.configure({
        Auth: {
            identityPoolId: 'XX-XXXX-X:XXXXXXXX-XXXX', // Amazon Cognito Identity Pool ID
            region: 'XX-XXXX-X', // Amazon Cognito Region
        }
    });

    // Call Auth.signIn with user credentials
    Auth.signIn(username, password)
        .then(user => console.log(user))
        .catch(err => console.log(err));
    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#imagePreviewResult').css('background-image', 'url(/get_image?p=' + data + ')');
                $('#imagePreviewResult').hide();
                $('.image-section-result').show();
                $('#imagePreviewResult').fadeIn(650); 
                console.log('Success!');
            },
        });
    });

});
