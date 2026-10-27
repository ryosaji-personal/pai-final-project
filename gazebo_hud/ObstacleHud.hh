#ifndef OBSTACLE_HUD_HH_
#define OBSTACLE_HUD_HH_

#include <gz/gui/Plugin.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/stringmsg.pb.h>

#include <QString>

namespace ignition
{
namespace gui
{

class ObstacleHud : public Plugin
{
  Q_OBJECT

  Q_PROPERTY(
      QString statusText
      READ StatusText
      NOTIFY StatusChanged
  )

public:
  ObstacleHud();

  ~ObstacleHud() override = default;

  void LoadConfig(
      const tinyxml2::XMLElement *_pluginElem
  ) override;

  Q_INVOKABLE QString StatusText() const;

signals:
  void StatusChanged();

private:
  void OnMessage(
      const gz::msgs::StringMsg &_msg
  );

  QString statusText{"Waiting for CNN..."};

  gz::transport::Node node;
};

}
}

#endif