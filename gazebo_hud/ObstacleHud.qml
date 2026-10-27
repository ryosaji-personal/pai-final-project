import QtQuick 2.12

Rectangle {
    anchors.fill: parent

    color: "#B0202835"
    radius: 8

    Text {
        id: titleText

        text: "AI OBSTACLE AVOIDANCE"

        x: 16
        y: 12

        color: "#66DDFF"
        font.pixelSize: 17
        font.bold: true
    }

    Rectangle {
        x: 16
        y: 42

        width: parent.width - 32
        height: 1

        color: "#607080"
    }

    Text {
        id: statusText

        x: 16
        y: 55

        text: ObstacleHud.statusText

        color: "white"

        font.family: "DejaVu Sans Mono"
        font.pixelSize: 16

        lineHeight: 1.35
    }
}