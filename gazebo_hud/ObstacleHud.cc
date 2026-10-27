#include "ObstacleHud.hh"

#include <QMetaObject>
#include <Qt>

#include <gz/plugin/Register.hh>

using namespace ignition;
using namespace gui;

ObstacleHud::ObstacleHud()
    : Plugin()
{
}

void ObstacleHud::LoadConfig(
    const tinyxml2::XMLElement *)
{
    const bool result = this->node.Subscribe(
        "/debug/status",
        &ObstacleHud::OnMessage,
        this
    );

    if (!result)
    {
        this->statusText = "Subscription failed";
        emit this->StatusChanged();
    }
}

QString ObstacleHud::StatusText() const
{
    return this->statusText;
}

void ObstacleHud::OnMessage(
    const gz::msgs::StringMsg &_msg)
{
    const QString value =
        QString::fromStdString(_msg.data());

    // Transportの受信スレッドから
    // GUIスレッドへ安全に更新を渡す
    QMetaObject::invokeMethod(
        this,
        [this, value]()
        {
            this->statusText = value;
            emit this->StatusChanged();
        },
        Qt::QueuedConnection
    );
}

IGNITION_ADD_PLUGIN(
    ObstacleHud,
    gui::Plugin
)