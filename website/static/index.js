function deleteImage(imageId) {
    fetch('/delete-image', {
        method: 'POST',
        body: JSON.stringify({ imageId: imageId}),
    }).then(() => {
        window.location.href = "/";
    })
}

function deleteFile(fileId) {
    fetch('/delete-file', {
        method: 'POST',
        body: JSON.stringify({ fileId: fileId}),
    }).then(() => {
        window.location.href = "/";
    })
}
